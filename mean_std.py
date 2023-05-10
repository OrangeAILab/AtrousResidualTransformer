import numpy as np
import cv2
import os
 
img_h, img_w = 224, 224
means, stdevs = [], []
img_list = []
 
# imgs_path = ''
# imgs_path_list = os.listdir(imgs_path)
 
# len_ = len(imgs_path_list)
# i = 0
# for item in imgs_path_list:
#     img = cv2.imread(os.path.join(imgs_path,item))
#     img = cv2.resize(img,(img_w,img_h))
#     img = img[:, :, :, np.newaxis]
#     img_list.append(img)
#     i += 1
#     # print(i,'/',len_)

imgs_path = 'ART/dataset/COVID-19 Radiography Database/NORMAL'
imgs_path_list = os.listdir(imgs_path)
 
len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    img = cv2.imread(os.path.join(imgs_path,item))
    img = cv2.resize(img,(img_w,img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)
    i += 1
    # print(i,'/',len_)

# print(len(img_list))

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.
 
for i in range(3):
    pixels = imgs[:, :, i, :].ravel() 
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))
 
# BGR --> RGB
means.reverse()
stdevs.reverse()
 
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))