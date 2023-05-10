###############################ART Dataloader######################
import os
import cv2
import numpy as np
import paddle
import random
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.vision import transforms as T
from matplotlib import pyplot as plt
import  random

class MyDataSet(Dataset):

    def __init__(self, mode='train', transform=None):
        '''
        define dataloader in ART
        '''
        super(MyDataSet, self).__init__()
        self.mode = mode

        # train dataset
        if self.mode == 'train':
            # train dataset root path
            
            TRAIN_IMAGE_DIR = 'ART/dataset/COVID-19 Radiography Database'
            TRAIN_IMAGE_LIST = 'ART/dataset/train_list.txt'
            self.train_image_dir = TRAIN_IMAGE_DIR
            self.train_image_list = TRAIN_IMAGE_LIST

            # data transform 
            self.transform = transform

            self.data_list = self.read_list()  
            print("train length:", len(self.data_list), "example:", self.data_list[0])

        # eval dataset
        if self.mode == 'eval':
            # eval dataset root path
            EVAL_IMAGE_DIR = 'ART/dataset/COVID-19 Radiography Database'
            EVAL_IMAGE_LIST = 'ART/dataset/val_list.txt'
            self.eval_image_dir = EVAL_IMAGE_DIR
            self.eval_image_list = EVAL_IMAGE_LIST

            self.transform = transform


            self.data_list = self.read_list() 
            print("eval length:", len(self.data_list), "example:", self.data_list[0])
        # test dataset
        if self.mode == 'test':
            self.data_list = []
            TEST_IMAGE_PATH = 'ART/dataset/COVID-19 Radiography Database'
            self.test_image_path = TEST_IMAGE_PATH
            for image in os.listdir(self.test_image_path):
                self.data_list.append(os.path.join(self.test_image_path, image))

    # generate data 
    def read_list(self):
        data_list = [] 
        if self.mode == 'train':
            image_dir = self.train_image_dir
            image_list = self.train_image_list
        else:
            image_dir = self.eval_image_dir
            image_list = self.eval_image_list

        with open(image_list) as f:
            for line in f:
                data_path = os.path.join(image_dir, line.split('|')[0])
                
                label = line.split('|')[1].strip('\n')
                data_list.append((data_path, label))
        random.shuffle(data_list)
        return data_list  
    def __getitem__(self, index):

        if self.mode == 'test':
            for data_path in self.data_list:
                # data = cv2.imread(data_path, cv2.IMREAD_COLOR)
                # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                data = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
                data = cv2.resize(data, (480, 480))
                data.astype(np.float32)
            return data

        else:
            data_path = self.data_list[index][0]
            # data = cv2.imread(data_path, cv2.IMREAD_COLOR)
            # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            data = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
            data = cv2.resize(data, (480, 480))

            label = self.data_list[index][1]
            # label = [int(label)]
            label = np.array(label, dtype='int64')
            # print( 'data.shape:', data.shape, 'label.shape=', label.shape)

            if self.transform:
                data = self.transform(data)

            data = data.astype(np.float32)
            return data, label

    def __len__(self):
        return len(self.data_list)


    def show(self):
        _, ax = plt.subplots(3, 3, figsize=(12, 12))
        images = []
        for data_path in self.data_list[:9]:
            print("data_path:", data_path)
            data = data_path[0]
            label = data_path[1]
           
            data = cv2.imread(data, cv2.IMREAD_COLOR)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            # data = cv2.resize(data,(512,512))
            images.append((data, label))

        for i, image in enumerate(images[:9]):
            ax[i // 3, i % 3].imshow(image[0])
            ax[i // 3, i % 3].axis('off')
            ax[i // 3, i % 3].set_title('label:'+image[1] )
        plt.show()
