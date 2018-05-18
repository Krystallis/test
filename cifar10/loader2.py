import csv
import os
import time
import cv2
import numpy as np
import re


def image_load(path):
    start = time.time()
    file_list = os.listdir(path)
    file_name = []
    for i in file_list:
        a = int(re.sub('[^0-9]','',i))
        file_name.append(a)
    file_name.sort()
    data = []
    for i in file_name:
        file = path + str(i) + '.png'
        data.append(file)
    end = time.time()
    print('image load time: %.2f' % float(end - start))

    return np.array(data)


def label_load(path):
    start = time.time()
    file = open(path)
    labeldata = csv.reader(file)
    labellist = []
    for i in labeldata:
        labellist.append(i)
    label = np.array(labellist)
    end = time.time()
    print('label load time: %.2f' % float(end - start))
    label = label.astype(int)
    label = np.eye(10)[label]
    label = np.squeeze(label,axis=1)
    return label


def next_batch(data_list,label,idx,batch_size):

    batch1 = data_list[idx * batch_size:idx * batch_size + batch_size]
    data = []
    for i in batch1:
        file = cv2.imread(i)
        file = np.divide(np.subtract(file, np.min(file)),np.subtract(np.max(file), np.min(file)))
        data.append(file)
    data = np.array(data)

    label2 = label[idx * batch_size:idx * batch_size + batch_size]

    index = np.arange(len(batch1))
    np.random.shuffle(index)
    data = data[index]
    label = label2[index]

    return data, label
