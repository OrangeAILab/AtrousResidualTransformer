'''
This is a backbone of  Transformer
'''
import  paddle
from  paddle import  nn
import  paddle.nn.functional as F
from  paddle.vision.models import resnet50





class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self,input):
        return input

#to change tensor from 2D to 1D in transformer layers, we need this function
def rearrange(x,string,l=0,w=0,h=0,**kwargs):
    b, n , c = x.shape[:3]

    if string == 'b n (h d) -> b h n d':
        x = paddle.reshape(x=x,shape=(b,n,h,-1)).transpose((0,2,1,3))

    if string == 'b (l w) n -> b n l w':

        x = paddle.reshape(x=x,shape=(b,l,w,-1)).transpose((0,3,1,2))
    if string == 'b (h d) l w -> b h (l w) d':
        b ,h_d , l,w = x.shape
        x = paddle.reshape(x=x,shape=(b,h,l*w,-1))

    return x


#Dilation projection block
class DilationProject(nn.Layer):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=2,dilation=2):
        super(DilationProject, self).__init__()

        self.dilationProject = nn.Sequential(
            nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ),
            nn.BatchNorm(num_channels=out_channels),
            nn.ReLU()
        )

    def forward(self,input):
        x = self.dilationProject(input)
        return x


#Define the Depth_Separable_Wise Conv2D
class DepthSepWiseConv2D(nn.Layer):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1):
        super(DepthSepWiseConv2D, self).__init__()

        #depthwise Conv2D
        self.depthwise = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels
        )

        #batch norm
        self.bn = nn.BatchNorm(num_channels=in_channels)

        #pointwise Conv2D
        self.pointwise = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1
        )

    def forward(self,input):
        x = self.depthwise(input)
        x = self.bn(x)
        x = self.pointwise(x)
        return  x


#Residual module in transformer
class Residual(nn.Layer):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn

    def forward(self,input,**kwargs):
        x = self.fn(input,**kwargs)
        return (x+input)


#Layer Norm plus
class PreNorm(nn.Layer):
    def __init__(self,dim,fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self,input, **kwargs):
        return self.fn(self.norm(input),**kwargs)



class FeedForward(nn.Layer):
    def __init__(self,dim,hidden_dim,dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=dim,out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim,out_features=dim),
            nn.Dropout(dropout)
        )
    def forward(self,input):
        return self.net(input)

'''
In ART, we use the DilationProject block to produce the q,k,v  instead of using Linear Project in ViT
'''
class ConvAttention(nn.Layer):
    def __init__(self,dim,img_size,heads=8,dim_head=64,kernel_size=3,q_stride=1,k_stride=1,v_stride=1,dropout=0.,
                 last_stage=False):
        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** (-0.5)
        pad = (kernel_size - q_stride) // 2
      
        self.to_q = DilationProject(in_channels=dim,out_channels=inner_dim,kernel_size=kernel_size,stride=q_stride,
                                    padding=3,dilation=3)
        self.to_k = DilationProject(in_channels=dim,out_channels=inner_dim,kernel_size=kernel_size,stride=k_stride,
                                    padding=3,dilation=3)
        self.to_v = DilationProject(in_channels=dim,out_channels=inner_dim,kernel_size=kernel_size,stride=v_stride,
                                    padding=3,dilation=3)

        self.to_out = nn.Sequential(
            nn.Linear(
                in_features= inner_dim,
                out_features= dim
            ),
            nn.Dropout(dropout)
        ) if project_out else Identity()


    def forward(self,x):
        #x : torch.Size([1, 3136, 64])
        b, n , c, h = *x.shape,self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            #cls_token = rearrange(paddle.unsqueeze(cls_token,axis=1),'b n (h d) -> b h n d', h = h)
            cls_token = rearrange(x=paddle.unsqueeze(cls_token,axis=1),string='b n (h d) -> b h n d',h=h)

        #x = rearrange(x,'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)#torch.Size([1, 64, 56, 56])
        x = rearrange(x=x,string='b (l w) n -> b n l w',l=self.img_size,w=self.img_size)

        q = self.to_q(x) #torch.Size([1, 64, 56, 56])
        #q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h) ##torch.Size([1, 1, 3136, 64])
        q = rearrange(x=q,string='b (h d) l w -> b h (l w) d',h=h)
        #q = q.reshape((b,1,))



        k = self.to_k(x)
        #k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        k = rearrange(x=k,string='b (h d) l w -> b h (l w) d',h=h)

        v = self.to_v(x)
        #v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(x=v,string='b (h d) l w -> b h (l w) d',h=h)

        if self.last_stage:
            q = paddle.concat((cls_token,q),axis=2)
            v = paddle.concat((cls_token,v),axis=2)
            k = paddle.concat((cls_token,k),axis=2)


        #calculate attention by matmul + scale
        attention = (q.matmul(k.transpose((0,1,3,2)))) * self.scale

        #pass softmax
        attention = F.softmax(attention,axis=-1)

        #matmul v
        out = (attention.matmul(v)).transpose((0,2,1,3)).reshape((b,n,c))

        #linear project
        out = self.to_out(out)
        return  out







