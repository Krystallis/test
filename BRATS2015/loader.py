import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

def data_list_load(path, image_type):
    OT = []
    image_type = image_type + '.'
    data = []

    for (path, dir, files) in os.walk(path):
        i = 0
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg':
                file = path+'\\'+str(i)+'.jpg'
                i+=1

                if image_type in file:
                    data.append(file)
                elif 'OT.' in file:
                    OT.append(file)

    return data,OT

def next_batch(data_list,label,idx,batch_size):
    data_list = np.array(data_list)
    label = np.array(label)

    batch1 = data_list[idx * batch_size:idx * batch_size + batch_size]
    label2 = label[idx * batch_size:idx * batch_size + batch_size]

    index = np.arange(len(batch1))
    np.random.shuffle(index)
    batch1 = batch1[index]
    label2 = label2[index]

    return batch1, label2

def data_shuffle(data,label):
    data = np.array(data)
    label = np.array(label)

    index = np.arange(len(data))
    np.random.shuffle(index)

    data = data[index]
    label = label[index]

    return data, label


def TT_split(data,label):

    trainX, testX, trainY, testY = train_test_split(data,label,test_size=0.15)

    return trainX, testX, trainY, testY


def read_image_grey_resized(data_list):

    if type(data_list) == list:
        data_list = data_list
    elif type(data_list) == str:
        data_list = [data_list]

    data = []
    for file in data_list:
        img = cv2.imread(file,0)
        img = cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
        if np.max(img) != 0:
            img = np.divide(np.subtract(img, np.min(img)),np.subtract(np.max(img), np.min(img)))

        data.append(img)

    return np.array(data).reshape([-1,224,224,1])

def read_label_grey_resized(data_list):

    if type(data_list) == list:
        data_list = data_list
    elif type(data_list) == str:
        data_list = [data_list]

    data = []
    for file in data_list:
        img = cv2.imread(file, 0)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)


        # 픽셀값 50을 기준으로 삼아 넘으면 1로 아니면 0으로 픽셀값을 변경
        img1 = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY)[1]
        # 위와는 반전되는 이미지를 생성
        img2 = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY_INV)[1]

        # (224,224,1)의 shape를 가지도록 reshape
        img1 = img1.reshape([224,224,1])
        img2 = img2.reshape([224,224,1])

        # 채널을 기준으로 concatnate
        img = np.concatenate((img1,img2),axis=2)

        data.append(img)

    return np.array(data).reshape([-1, 224, 224, 2])
