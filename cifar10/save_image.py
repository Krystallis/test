import os
import loader2
import numpy as np
import cv2
import csv

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train_path = 'C:\\project\\data\\cifar10\\train\\'
test_path = 'C:\\project\\data\\cifar10\\test\\'
tr_list = os.listdir(train_path)
te_list = os.listdir(test_path)
print(tr_list)
print(te_list)


image = []

for i in tr_list:
    file = train_path + i
    print(file)
    data = unpickle(file)
    la = data[b'data']
    for i in la:
        image.append(i)

file = np.array(image)
print(file.shape)



image = []

for i in file:
    R = i[0:1024].reshape([32,32,1])
    G = i[1024:2048].reshape([32,32,1])
    B = i[2048:3072].reshape([32,32,1])
    img = np.concatenate([B,G,R],axis=2)
    image.append(img)

image = np.array(image)

path = 'E:/cifar10/train/'

for i in range(len(image)):
    filename = path + str(i+1) + '.png'
    print(filename)
    cv2.imwrite(filename,image[i])

