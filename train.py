import cv2
from PIL import Image
import imageio
from tqdm import tqdm

import torch
import torch.nn.init
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import json
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import time

import util
import NeuralNets as NN
import dataset

import os

PRE_TRAIN_PATH = "./coco_data/train_2017/img/"
PRE_LABEL_PATH = "./coco_data/train_2017/label.json"

TRAIN_PATH = "./second_train_data/img/"
LABEL_PATH = "./second_train_data/label.json"

class FocusDataset(Dataset):
    def __init__(self, path, label_path, transform=None):
        self.x_datas = []
        self.y_datas = {}
        self.path = path
        self.transform = transform

        fileDir = os.listdir(path)
        for file in fileDir:
            self.x_datas.append(file)
        
        file = open(label_path, 'r')
        c = json.load(file)
        for each in c:
            self.y_datas[each["img_path"]] = each["focus_point"]

    def __len__(self):
        return len(self.x_datas)

    def __getitem__(self, idx):
        target = self.x_datas[idx]
        
        x = Image.open(self.path + target).convert("RGB")

        ori_shape = x.size
        y = self.y_datas[target]
        y = util.point_adjust(y, ori_shape, NN.INPUT_SHAPE)
        y = util.pointMap(y)

        if self.transform:
            x = util.transform(x, y)

        x = x.resize(NN.INPUT_SHAPE)
        x = np.array(x)
        return x , y


preTrainDataset = FocusDataset(PRE_TRAIN_PATH, PRE_LABEL_PATH)
trainDataset = FocusDataset(TRAIN_PATH, LABEL_PATH)










import random

fig, ax = plt.subplots(4, 5)


for i in range(5):
    randId = int(random.random() * len(trainDataset))
    preTrainData = preTrainDataset[randId]
    trainData = trainDataset[randId]


    preImg = preTrainData[0]
    preFocus = preTrainData[1] * 255
    img = trainData[0]
    focus = trainData[1] * 255

    ax[0, i].imshow(preImg)
    ax[1, i].imshow(focus)
    #ax[1, i].scatter(preFocus[:, 0], preFocus[:, 1], marker="o", color='green')
    ax[1, i].set_title("train {}".format(i))
    
    ax[2, i].imshow(img)
    ax[3, i].imshow(focus)
    #ax[3, i].scatter(focus[:, 0], focus[:, 1], marker="o", color='green')
    ax[3, i].set_title("train {}".format(i))


plt.tight_layout()
plt.show()