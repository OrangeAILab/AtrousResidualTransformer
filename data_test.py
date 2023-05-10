# coding=gbk
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
parser.add_argument('--lr', default=0.001, type=float, help='learning_rate')
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
    T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
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
    T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
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

