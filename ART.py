'''
This is the source code of ART, DilationResidualTransformer, which combine the CNN local features and Transformer
global information perfectly.


'''



import  paddle
import  paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal,Constant
from backbones.convolutionalTransformer import ConvAttention,PreNorm,FeedForward,DilationProject
from  backbones.resnet import  ResNet,BasicBlock,BottleneckBlock
#from backbones.resnet_vd import ResNet_vd
import  paddle.nn.functional as F

# make the data subject to random truncated Gaussian distribution !
truncatedNormalInitial = TruncatedNormal(std=.02)
zero = Constant(value=0.)
one = Constant(value=1.)




#Reshape Layers
class Rearrage(nn.Layer):
    def __init__(self,string,h,w):
        super().__init__()
        self.string = string
        self.h = h
        self.w = w
    def forward(self,input):

        if self.string == 'b c h w -> b (h w) c':
            N, C, H, W = input.shape
            x = paddle.reshape(x=input,shape=(N,-1,self.h*self.w)).transpose((0,2,1))

        if self.string == 'b (h w) c -> b c h w':
            N,_,C = input.shape
            x = paddle.reshape(x=input,shape=(N,self.h,self.w,-1)).transpose((0,3,1,2))
        return x


#Transformer layers
class Transformer(nn.Layer):
    def __init__(self,dim,img_size,depth,heads,dim_head,mlp_dim,dropout=0.,last_stage=False):
        super().__init__()
        self.layers = nn.LayerList([
            nn.LayerList([
                PreNorm(dim=dim,fn=ConvAttention(dim,img_size,heads=heads,dim_head=dim_head,dropout=dropout,last_stage=last_stage)),
                PreNorm(dim=dim,fn=FeedForward(dim=dim,hidden_dim=mlp_dim,dropout=dropout))
            ]) for _ in range(depth)
        ])

    def forward(self,x):
        for attn,ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class DilationResidualTransformer(nn.Layer):
    def __init__(self, image_size=224, in_channels=3, num_classes=2, dim=64, kernels=[7, 3, 3, 3], strides=[4, 2, 2, 2],
                 heads=[1, 2, 4, 8], depth=[1, 1, 1, 1], pool=True, dropout=0., scale_dim=4):
        super().__init__()

        self.image_size = image_size
        self.dim = dim
        self.pool = pool
        self.num_classes = num_classes

        ##################################stage one#####################################
        self.stage1_dilation_embed = nn.Sequential(
            # nn.Conv2D(
            #     in_channels=in_channels,
            #     out_channels=dim,
            #     kernel_size=kernels[0],
            #     stride=strides[0],
            #     padding=2
            # ),
            DilationProject(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=kernels[0],
                stride=strides[0],
                padding=5,
                dilation=2
            ),

            Rearrage('b c h w -> b (h w) c', h=image_size // 4, w=image_size // 4),

            nn.LayerNorm(dim)
        )
        self.stage1_transformer = nn.Sequential(
            Transformer(
                dim=dim,
                img_size=image_size // 4,
                depth=depth[0],
                heads=heads[0],
                dim_head=self.dim,
                mlp_dim=dim * scale_dim,
                dropout=dropout
            ),

            Rearrage(string='b (h w) c -> b c h w', h=image_size // 4, w=image_size // 4)
        )



        ##################################stage Second#####################################

        in_channels = dim #64
        scale = heads[1] //heads[0] #expand twice times
        dim = scale * dim #128
        #Conv2D(64, 128, 3, 2, 1)



        self.stage2_dilation_embed = nn.Sequential(
            # nn.Conv2D(
            #     in_channels=in_channels,
            #     out_channels=dim,
            #     kernel_size=kernels[1],
            #     stride=strides[1],
            #     padding=1
            # ),
            DilationProject(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=kernels[1],
                stride=strides[1],
                padding=2,
                dilation=2
            ),

            Rearrage(string='b c h w -> b (h w) c', h=image_size // 8, w=image_size // 8),
            nn.LayerNorm(dim)
        )
        self.stage2_transformer = nn.Sequential(
            Transformer(
                dim=dim,
                img_size=image_size // 8,
                depth=depth[1],
                heads=heads[1],
                dim_head=self.dim,
                mlp_dim=dim * scale_dim,
                dropout=dropout
            ),
            Rearrage(string='b (h w) c -> b c h w', h=image_size // 8, w=image_size // 8)
        )

        ##################################stage Thrid#####################################
        in_channels = dim# 128
        scale = heads[2] // heads[1]
        dim = scale * dim #256
        #conv2d (128,256,3,2,1 )
        self.stage3_dilation_embed = nn.Sequential(
            # nn.Conv2D(
            #     in_channels = in_channels,
            #     out_channels = dim,
            #     kernel_size=kernels[2],
            #     stride=strides[2],
            #     padding=1
            # ),
            DilationProject(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=kernels[2],
                stride=strides[2],
                padding=2,
                dilation=2
            ),
            Rearrage(string='b c h w -> b (h w) c', h=image_size // 16, w=image_size // 16),
            nn.LayerNorm(dim)
        )

        self.stage3_transformer = nn.Sequential(
            Transformer(
                dim=dim,
                img_size=image_size//16,
                depth=depth[2],
                heads=heads[2],
                dim_head=self.dim,
                mlp_dim=dim*scale_dim,
                dropout=dropout
            ),
            Rearrage(string='b (h w) c -> b c h w', h=image_size // 16, w=image_size // 16)
        )

        ##################################stage fourth#####################################
        in_channels = dim # 256
        scale = heads[3] // heads[2]
        dim = scale * dim # 512
        #conv2d(256,512,3,2,1 )
        self.stage4_dilation_embed = nn.Sequential(
            # nn.Conv2D(
            #     in_channels = in_channels,
            #     out_channels = dim,
            #     kernel_size = kernels[3],
            #     stride = strides[3],
            #     padding = 1
            # ),
            DilationProject(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=kernels[3],
                stride=strides[3],
                padding=2,
                dilation=2
            ),
            Rearrage('b c h w -> b (h w) c', h=image_size // 32, w=image_size // 32),
            nn.LayerNorm(dim)
        )

        self.stage4_transformer = nn.Sequential(
            Transformer(
                dim=dim,
                img_size=image_size // 32,
                depth=depth[3],
                heads=heads[3],
                dim_head=self.dim,
                mlp_dim=dim * scale_dim,
                dropout=dropout
            ),
            Rearrage(string='b (h w) c -> b c h w', h=image_size // 32, w=image_size // 32)
        )

        #enhance block for transformer
        self.stage1_dilation_conv2D= DilationProject(in_channels=64,out_channels=64, kernel_size=3,stride=1,padding=2,dilation=2)

        self.stage2_dilation_conv2D= DilationProject(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2,dilation=2)

        self.stage3_dilation_conv2D = DilationProject(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2,dilation=2)

        self.stage4_dilation_conv2D= DilationProject(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2,dilation=2)

        #concat the local head and global head
        self.num_channels = 512
        #conv1x1 reduce the channel dim

        #classification head
        if self.pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1,1))

        if num_classes > 0:
            self.fc = nn.Linear(in_features=self.num_channels,out_features=self.num_classes)

        self.drop_out = nn.Dropout(p=0.5)

        '''
        Ablation Study :
        1.Add Posotion Embedding
        2.Add Class Token
        '''
        self.pos_embedding = self.create_parameter(
            shape=(1,3136,64),
            default_initializer=zero
        )
        self.add_parameter('pos_embedding',self.pos_embedding)

        self.cls_token = self.create_parameter(
            shape=(1,1,dim),
            default_initializer=zero
        )
        self.add_parameter('cls_token',self.cls_token)



    def forward(self,input):
        #CNN head forward  ->>>>>>>>>>>>>>>>>>>>> (N,3,224,224)

        # res is the global feature map(N,64,56,56) - > get into a residual block
        x = self.stage1_dilation_embed(input)
        #x = x + self.pos_embedding
        res = self.stage1_transformer(x)

        x = self.stage1_dilation_conv2D(res)
        x = self.stage1_dilation_conv2D(x)
        x = paddle.add(x=x,y=res)

        #x = self.drop_out(x)


        # res is the global feature map(N,128,28,28) - > get into a residual block
        x = self.stage2_dilation_embed(x)
        res = self.stage2_transformer(x)

        x = self.stage2_dilation_conv2D(res)
        x = self.stage2_dilation_conv2D(x)
        x = paddle.add(x=x,y=res)

        #x = self.drop_out(x)

        # res is the global feature map(N,256,14,14) - > get into a residual block
        x = self.stage3_dilation_embed(x)
        res = self.stage3_transformer(x)

        x = self.stage3_dilation_conv2D(res)
        x = self.stage3_dilation_conv2D(x)
        x = paddle.add(x=x,y=res)

        #x = self.drop_out(x)

        # res is the global feature map(N,512,7,7) - > get into a residual block
        x = self.stage4_dilation_embed(x)
        res = self.stage4_transformer(x)

        x = self.stage4_dilation_conv2D(res)
        x = self.stage4_dilation_conv2D(x)
        x = paddle.add(x=x,y=res)

        #x = self.drop_out(x)

        if self.pool:
            out = self.avgpool(x) #(N,512,7,7)->(N,512,1,1)

        if self.num_classes > 0:
            out = paddle.flatten(out,1)#(N,512,1,1) -> (N,512X1X1)
            out = self.fc(out)
            #out = F.softmax(out)

        return out







if __name__ == '__main__':

    model = DilationResidualTransformer(image_size=224,in_channels=3,num_classes=2)
    #model = ResNet(BottleneckBlock,152,num_classes=2)
    #model = ResNet(BasicBlock,18,num_classes=2)
    paddle.summary(net=model,input_size=(1,3,224,224))

    out = model(paddle.randn(shape=(1,3,224,224)))
    print('out :',out,'out.shape:,',out.shape)


    FLOPs = paddle.flops(model,[1,3,224,224],print_detail=True)
    print(FLOPs)
