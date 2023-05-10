
import argparse
import paddle
from PIL import Image
import paddle.vision.transforms as T
#from dataloader import  CovidCTDataset
from dataloader import   MyDataSet
from paddle.io import DataLoader
from matplotlib import pyplot as plt

from ART import  DilationResidualTransformer

from backbones.resnet_ import _resnet,BottleneckBlock,BasicBlock
from backbones.resnet import ResNet
from backbones.densenet import  DenseNet
from backbones.resnet50_vd import  ResNet_vd
from backbones.VGG import  VGG16,VGG19

from backbones.efficientnet import  EfficientNetB0,EfficientNetB1,EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6

from backbones.vision_transformer import VisionTransformer, ViT_small_patch16_224,ViT_base_patch16_224,ViT_large_patch16_224,ViT_huge_patch16_224
from backbones.distilled_vision_transformer import DeiT_tiny_patch16_224,DeiT_small_patch16_224,DeiT_base_patch16_224,DeiT_tiny_distilled_patch16_224,DeiT_small_distilled_patch16_224,DeiT_base_distilled_patch16_224


from paddle.vision.datasets import  Cifar10,Flowers

########################################set the super parameters########################################################
parser = argparse.ArgumentParser(description="train set")
parser.add_argument('--image_size', default=224, type=int, help='the input size')
parser.add_argument('--class_dim', default=3, type=int, help='the classification dims')
parser.add_argument('--pretrained', default=False, type=bool, help='whether load the pretrained weights')
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--lr', default=0.003, type=float, help='learning_rate')
parser.add_argument('--weight_decay', default=0.005, type=float, help='the weight_decay rate')
parser.add_argument('--epochs', default=300, type=int, help='the train epochs')
parser.add_argument('--warmup', default=20, type=int, help='the warmup of learning_rate')
parser.add_argument('--weights',default='./vit_small_patch16_224.pdparams',type=str,help='the path of pretained model')
args = parser.parse_args()

########################################################################################################################

########################################################################################################################


##################################################################################################

##################################################################################################

############################################Data Augumention########################################
#this data distribution :normMean = [0.5952323, 0.5948437, 0.59467155] normStd = [0.32732046, 0.32735404, 0.32730502]

train_transforms = T.Compose([
    T.Resize((224, 224)),
    #T.RandomResizedCrop(32),
    #T.RandomHorizontalFlip(),
    #T.RandomRotation(180),
    #T.ColorJitter(),
    # T.ToTensor(),
    # T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    T.Normalize(mean=[0.47951055, 0.47951055, 0.47951055],std=[0.24667357, 0.24667357, 0.24667357]),
  
    #T.Normalize(mean=[127.5],std=[127.5],data_format='CHW'),
    #T.Normalize(mean=[0.5952323, 0.5948437, 0.59467155],std=[0.32732046, 0.32735404, 0.32730502]),
    T.Transpose(order=(0, 2, 1))
])

# train_transforms = T.Compose([
#     T.Resize(224),
#     T.RandomResizedCrop((224), scale=(0.5, 1.0)),
#     T.RandomHorizontalFlip(),
#     T.RandomRotation(180),
#     T.CenterCrop(224),
#     T.RandomVerticalFlip(),
#     T.BrightnessTransform(0.4),
#     T.SaturationTransform(0.4),
#     T.ContrastTransform(0.4),
#     T.HueTransform(0.4),
#     T.ColorJitter(0.4, 0.4, 0.4, 0.4),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
#     T.Transpose(order=(0, 2, 1))
# ])



eval_transforms = T.Compose([
    T.Resize((224, 224)),
    #T.RandomResizedCrop(224),
    # T.ToTensor(),
    #T.Normalize(mean=[127.5],std=[127.5],data_format='CHW'),
    # T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    T.Normalize(mean=[0.47951055, 0.47951055, 0.47951055],std=[0.24667357, 0.24667357, 0.24667357]),
  
    #T.Normalize(mean=[0.5952323, 0.5948437, 0.59467155],std=[0.32732046, 0.32735404, 0.32730502]),
    T.Transpose(order=(0, 2, 1))
])

#train_dataset = Cifar10(mode='train',transform=train_transforms)
#eval_dataset = Cifar10(mode='test',transform=eval_transforms)


train_dataset =  MyDataSet(mode='train',transform=train_transforms)
eval_dataset = MyDataSet(mode='train',transform=eval_transforms)

train_dataset_loader = DataLoader(
    dataset=train_dataset,
    places=paddle.CUDAPlace(0),
    batch_size=32,
    shuffle=True
)

eval_dataset_loader = DataLoader(
    dataset=eval_dataset,
    places=paddle.CUDAPlace(0),
    batch_size=32,
    shuffle=False
)
print("data augument is successfull")





##################################################################################################

##################################################################################################

print('========================check train dataset informatio===============================')
print("train length:",len(train_dataset),"")

#train_dataset.show()
for image,label in train_dataset:
    print('check data shape:image.shape:{}[C,H,W],label.shape:{}'.format(image.shape,label.shape))
    print('check data value image value:{},label.value:{}'.format(image,label))
    break

print("a batch data information")

for batch_id,data in enumerate(train_dataset_loader()):
    x_data = data[0]
    y_data = data[1]
    print("a batch data shape[N,C,H,W]:",x_data.numpy().shape)
    print("a batch label shape",y_data.numpy().shape)
    break


print('========================check eval dataset informatio===============================')
print("eval dataset:",len(eval_dataset))
#print("-------->:")
#eval_dataset.show()
for image,label in eval_dataset:
    print('check data shape image.shape:{}[C,H,W],label.shape:{}'.format(image.shape,label.shape))
    print('check data value image value:{},label.value:{}'.format(image,label))
    break


print("a batch data information")

for batch_id,data in enumerate(eval_dataset_loader()):
    x_data = data[0]
    y_data = data[1]
    print("batch shape[N,C,H,W]:",x_data.numpy().shape)
    print("batch shape",y_data.numpy().shape)
    break

print("dataloader is successfully finished !")



##################################################################################################

##################################################################################################


# Set trainning parameters
base_lr = args.lr
warmup_setps = args.warmup
weight_decay = args.weight_decay
epochs = args.epochs
image_size = args.image_size
class_dim = args.class_dim


# save bset model
class SaveBestModel(paddle.callbacks.Callback):
    def __init__(self, target=1, path='./out/save_best_model', verbose=0):
        self.target = target
        self.epoch = None
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch

    def on_eval_end(self, logs=None):
        if logs.get('loss')[0] < self.target:
            self.target = logs.get('loss')[0]
            self.model.save(self.path)
            print('best model is loss {} at epoch {}'.format(self.target, self.epoch))


def make_optimizer_momentum(parameters=None):
    momentum = 0.9  
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=base_lr,
        T_max=epochs,
        verbose=False
    )
    learning_rate = paddle.optimizer.lr.LinearWarmup(
        learning_rate=learning_rate,
        warmup_steps=warmup_setps,
        start_lr=base_lr / 5.,
        end_lr=base_lr,
        verbose=False
    )
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=momentum,
        parameters=parameters
    )
    return optimizer


def make_optimizer_adam(parameters=None):
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=base_lr,
        T_max=epochs,
        verbose=False
    )

    learning_rate = paddle.optimizer.lr.LinearWarmup(
        learning_rate=learning_rate,
        warmup_steps=warmup_setps,
        start_lr=base_lr / 5.,
        end_lr=base_lr,
        verbose=False
    )

    optimizer = paddle.optimizer.Adam(
        learning_rate=learning_rate,
        parameters=parameters
    )

    return optimizer

callback_vdl = paddle.callbacks.VisualDL(log_dir='./Visual_DLs/VisualDl_log_dir')
callback2 = SaveBestModel(target=1, path='./out/save_best_model')


#model = _resnet('resnet50',BottleneckBlock,50,pretrained=False,num_classes=2)
print("============================ATTENTION ! You are successfully  using ResNet50 NetWork============================")
#model = _resnet('resnet18',BasicBlock,18,pretrained=True,num_classes=2)
#model = _resnet('resnet18',BasicBlock,18,pretrained=True)


#model = VGG16(class_dim=2)
#model = VGG19(class_dim=2)
#model = ResNet(layers=50,class_dim=10)
#model = ResNet(layers=101,class_dim=2)
#model = ResNet_vd(layers=50,class_dim=2)
#model = ResNet_vd(layers=18,class_dim=2)
#model = ResNet_vd(layers=101,class_dim=2)
#model = ResNet_vd(layers=152,class_dim=2)
#model = ResNet_vd(layers=200,class_dim=2)
#model = ViT_small_patch16_224(imgae_size=224,class_dim=2)
#model = ViT_base_patch16_224(image_size=224,class_dim=2)
#model = ViT_large_patch16_224(image_size=224,class_dim=2)
#model = ViT_huge_patch16_224(image_size=224,class_dim=2)
#model = DoubleHeadConvTransformer(image_size=224,in_channels=3,num_classes=2)
#model = DenseNet(layers=169,class_dim=2)
#model = DenseNet(layers=264,class_dim=2)
#model = DenseNet(layers=161,class_dim=2)

#model  = ViT_small_patch16_224(class_dim=2)
#model = DeiT_tiny_patch16_224(img_size=224,class_dim=2)
#model = DeiT_small_patch16_224(class_dim=2)
#model = DeiT_small_patch16_224(img_size=224,class_dim=2)
#model = DeiT_base_patch16_224(img_size=224,class_dim=2)
model = DilationResidualTransformer(image_size=224, num_classes=3)

#model = EfficientNetB0(class_dim=2)
#model = EfficientNetB1(class_dim=2)
#model = EfficientNetB2(class_dim=2)
#model = EfficientNetB3(class_dim=2)
#model = EfficientNetB4(class_dim=2)
#model = EfficientNetB5(class_dim=2)
#model = EfficientNetB6(class_dim=2)


paddle.summary(model,input_size=(1,3,224,224))
out = model(paddle.randn((1,3,224,224)))
print('out.shape:',out.shape)

if args.pretrained:
    model.set_state_dict(paddle.load(args.weigts))
    print("Loading the pretrained weights successfully!")



# set optimizer
#optimizer = make_optimizer_momentum(model.parameters())
#optimizer = make_optimizer_adam(model.parameters())
#optimizer = paddle.optimizer.Adam(learning_rate=0.0003,parameters=model.parameters())
optimizer = paddle.optimizer.Adam(learning_rate=0.001,parameters=model.parameters())

model1 = paddle.Model(model)

model1.prepare(
    optimizer = optimizer,
    loss=paddle.nn.CrossEntropyLoss(),
    #metrics=[paddle.metric.Accuracy(),paddle.metric.Precision(),paddle.metric.Recall()]
    metrics=[paddle.metric.Accuracy()]
)
model1.fit(
    train_data = train_dataset_loader,
    eval_data  = eval_dataset_loader,
    epochs = 20,
    callbacks = [callback_vdl,callback2],
    verbose = 1
)

eval_result = model1.evaluate(eval_dataset, verbose=1)
print("==================finally eval result :",eval_result)
print("model saving ---------》")
model1.save('./out/save_best_model')
paddle.jit.save(model1,'./out/inference')
