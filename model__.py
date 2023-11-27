import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1, 
                 normalize=True, dropout=False, p=0.2,
                 residual_connection=True):
        super().__init__()
        self.normalize = normalize
        self.dropout = dropout
        self.p = p
        self.residual_connection = residual_connection
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, 
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        if normalize:
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)
        if dropout:
            self.drop = nn.Dropout(p=p)
        
    def forward(self, X):
        Y = self.conv1(X)
        if self.normalize:
            Y = F.relu(self.bn1(Y))
            Y = self.bn2(self.conv2(Y))
        else:
            Y = F.relu(Y)
            Y = self.conv2(Y)
        if self.residual_connection:
            if self.conv3:
                X = self.conv3(X)
            Y += X
        Y = F.relu(Y)
        if self.dropout:
            Y = self.drop(Y)
        return Y
    
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False, normalize=True, dropout=False, 
                 p=0.2, residual_connection=True):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2, 
                                normalize=normalize, dropout=dropout, p=p, 
                                residual_connection=residual_connection))
            blk.append(Residual(num_channels, num_channels,
                                normalize=normalize, dropout=dropout, p=p, 
                                residual_connection=residual_connection))
    return blk

def CNN_net(channels=[64,128,256,512], normalize=True, 
            dropout=False, p=0.2, residual_connection=True):
    assert(channels[0] == 64)
    blk = []
    blk.append(nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                             nn.BatchNorm2d(64), nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)))
    blk.append(nn.Sequential(*resnet_block(64, 64, 2, first_block=True,
                                            normalize=normalize, dropout=dropout, 
                                            p=p, residual_connection=residual_connection)))
    for i in range(len(channels)-1):
        blk.append(nn.Sequential(*resnet_block(channels[i], channels[i+1], 2,
                                               normalize=normalize, dropout=dropout, 
                                               p=p, residual_connection=residual_connection)))

    net = nn.Sequential(*blk, nn.AdaptiveAvgPool2d((1,1)), 
                        nn.Flatten(), nn.Linear(channels[-1], 200))
    return net

def CNN_net(channels=[64,128,256,512], normalize=True, 
            dropout=False, p=0.2, residual_connection=True):
    blk = []
