import numpy as np
import NeuralNets as NN

import random
import torch
import torchvision.transforms.functional as FT

from scipy import ndimage

import cv2

COLORS = [(0,0,255), (0, 233, 233), (0, 233, 233), (255, 0, 0), (255, 0, 0)]


#https://guru.tistory.com/73
class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)

        else:
            assert len(output_size)==2 
            self.output_size = output_size

    def __call__(self, sample, point):
        image = sample
        new_point = point

        xs = point[:,0]
        ys = point[:,1]




        h, w = image.shape[:2]
        new = self.output_size

        rand_x = [[0, min(xs)/2], [(w+max(xs))/2, w]]
        rand_y = [[0, min(ys)/2], [(h+max(ys))/2, h]]
        top_idx = [np.random.randint(0, 2), np.random.randint(0, 2)]
        left_idx = [np.random.randint(0, 2), np.random.randint(0, 2)]

        top = np.random.randint(rand_x[top_idx[0]], rand_x[top_idx[1]])
        left = np.random.randint(rand_y[left_idx[0]], rand_y[left_idx[0]])

        image = image[top[0]:top[1], left[0]:left[1]]

        return image, new_point





def getFileNum(n):
    return str(n).zfill(5)  



def point_adjust(points, base, to):
    res = points
    res = res * np.asarray(to)/np.asarray(base)

    return res.astype(np.int)

def pointMap(points, size=(72, 56)):
    base = np.ones((size[1], size[0]))
    base *= -1
    for each in points:
        each[0] *= size[0]/NN.INPUT_SHAPE[0]
        each[1] *= size[1]/NN.INPUT_SHAPE[1]

        base[int(each[1]), int(each[0])] =  255

    return base

def draw_points(img, points, colors=COLORS, size=6):
    result_img = img.copy()
    i =0
    for each in points:
        result_img = cv2.line(result_img, tuple(each), tuple(each), colors[i], size)
        i += 1
        
    return result_img

def normalize(arr, _min=0, _max=255):
    arr = arr - np.min(arr) + _min
    if np.max(arr) == 0: 
        return arr
    arr = np.asarray(arr, dtype=np.float32) * _max/np.max(arr)
    return arr

def popPoints(arr, threshold=1):
    return arr[arr>threshold]

def cvt2Heatmap(img, superimposeOn=None, ratio=0.8, threshold=128):
    ori = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    ori[ori < threshold] = 0
    img = cv2.applyColorMap(ori, cv2.COLORMAP_JET)

    # color map 0 -> blue(128)
    # cut blue background
    blue_img = img[..., 0]
    blue_img[blue_img <= 128] = 0
    img[..., 0] = blue_img

    # impose on impose image
    if superimposeOn is not None:
        if superimposeOn.shape != img.shape:
            superimposeOn = cv2.resize(superimposeOn, (img.shape[1], img.shape[0]))
        img = cv2.addWeighted(superimposeOn, ratio, img, 0.6, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img