import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

# 自定义数据目录
data_path = '../data/tiny-imagenet-200/tiny-imagenet-200/'


# 定义数据类处理文件
class RawData:

    __labels_t_path = '%s%s' % (data_path, 'wnids.txt')
    __train_data_path = '%s%s' % (data_path, 'train/')
    __val_data_path = '%s%s' % (data_path, 'val/')

    __labels_t = None
    __image_names = None

    __val_labels_t = None
    __val_labels = None
    __val_names = None

    @staticmethod
    def labels_t():
        if RawData.__labels_t is None:
            labels_t = []
            with open(RawData.__labels_t_path) as wnid:
                for line in wnid:
                    labels_t.append(line.strip('\n'))

            RawData.__labels_t = labels_t

        return RawData.__labels_t

    @staticmethod
    def image_names():
        if RawData.__image_names is None:
            image_names = []
            labels_t = RawData.labels_t()
            for label in labels_t:
                txt_path = RawData.__train_data_path + label + '/' + label + '_boxes.txt'
                image_name = []
                with open(txt_path) as txt:
                    for line in txt:
                        image_name.append(line.strip('\n').split('\t')[0])
                image_names.append(image_name)

            RawData.__image_names = image_names

        return RawData.__image_names

    @staticmethod
    def val_labels_t():
        if RawData.__val_labels_t is None:
            val_labels_t = []
            with open(RawData.__val_data_path + 'val_annotations.txt') as txt:
                for line in txt:
                    val_labels_t.append(line.strip('\n').split('\t')[1])

            RawData.__val_labels_t = val_labels_t

        return RawData.__val_labels_t

    @staticmethod
    def val_names():
        if RawData.__val_names is None:
            val_names = []
            with open(RawData.__val_data_path + 'val_annotations.txt') as txt:
                for line in txt:
                    val_names.append(line.strip('\n').split('\t')[0])

            RawData.__val_names = val_names

        return RawData.__val_names

    @staticmethod
    def val_labels():
        if RawData.__val_labels is None:
            val_labels = []
            val_labels_t = RawData.val_labels_t()
            labels_t = RawData.labels_t()
            for i in range(len(val_labels_t)):
                for i_t in range(len(labels_t)):
                    if val_labels_t[i] == labels_t[i_t]:
                        val_labels.append(i_t)
            val_labels = np.array(val_labels)

            RawData.__val_labels = val_labels

        return RawData.__val_labels


# 定义 Dataset 类
class Data(Dataset):

    def __init__(self, type_, transform):
        """
        type_: 选择训练集还是验证集
        """
        self.__train_data_path = '%s%s' % (data_path, 'train/')
        self.__val_data_path = '%s%s' % (data_path, 'val/')

        self.type = type_

        self.labels_t = RawData.labels_t()
        self.image_names = RawData.image_names()
        self.val_names = RawData.val_names()

        self.transform = transform

    def __getitem__(self, index):
        label = None
        image = None

        labels_t = self.labels_t
        image_names = self.image_names
        val_labels = RawData.val_labels()
        val_names = self.val_names

        if self.type == "train":
            label = index // 500  # 每个类别的图片 500 张
            remain = index % 500
            image_path = os.path.join(self.__train_data_path, labels_t[label], 'images', image_names[label][remain])
            image = cv2.imread(image_path)
            image = np.array(image).reshape(64, 64, 3)

        elif self.type == "val":
            label = val_labels[index]
            val_image_path = os.path.join(self.__val_data_path, 'images', val_names[index])
            image = np.array(cv2.imread(val_image_path)).reshape(64, 64, 3)

        return label, self.transform(image)

    def __len__(self):
        len_ = 0
        if self.type == "train":
            len_ = len(self.image_names) * len(self.image_names[0])
        elif self.type == "val":
            len_ = len(self.val_names)

        return len_


class residual_block(nn.Module):
    def __init__(self, channel, normalize=False, dropout=False, 
                 p=0.2, residual_connection=True):
        super(residual_block, self).__init__()
        
        self.normalize = normalize
        self.dropout = dropout
        self.residual_connection = residual_connection
        
        self.cov1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.cov2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.relu = nn.ReLU()


        if normalize:
            self.nor1 = nn.BatchNorm2d(channel)
            self.nor2 = nn.BatchNorm2d(channel)

        if dropout:
            self.drop = nn.Dropout(p=p)

    def forward(self, X):
        Y = self.cov1(X)

        if self.normalize:
            Y = self.relu(self.nor1(Y))
            Y = self.cov2(Y)
            Y = self.relu(self.nor2(Y))
        else:
            Y = self.relu(Y)
            Y = self.cov2(Y)
            Y = self.relu(Y)

        if self.dropout:
            Y = self.drop(Y)

        if self.residual_connection:
            Y = X + Y
        
        return Y


class nor_cov(nn.Module):
    def __init__(self, in_channel, out_channel, 
                 normalize=False, dropout=False, p=0.2):
        super(nor_cov, self).__init__()

        self.normalize = normalize
        self.dropout = dropout

        self.cov = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.relu = nn.ReLU()
        
        if normalize:
            self.nor = nn.BatchNorm2d(out_channel)

        if dropout:
            self.drop = nn.Dropout(p=p)

    def forward(self, x):
        x = self.cov(x)

        if self.normalize:
            x = self.nor(x) 

        x = self.relu(x)

        if self.dropout:
            x = self.drop(x)

        return x


def CNN_net(normalize=False, residual_connection=True, 
            p=0.2, channels=[3, 64, 128], dropout=True):
    blk = []
    for i in range(len(channels)-1):
        in_channel = channels[i]
        out_channel = channels[i+1]
        blk.append(nor_cov(in_channel, out_channel,
                           normalize=normalize,
                           dropout=dropout, p=p))
        blk.append(residual_block(channel=out_channel,
                                  normalize=normalize,
                                  dropout=dropout, p=p,
                                  residual_connection=residual_connection))
        blk.append(nn.MaxPool2d(2,2))

    net = nn.Sequential(*blk, nn.AdaptiveAvgPool2d((1,1)), 
                        nn.Flatten(), nn.Linear(channels[-1], 200))
    return net