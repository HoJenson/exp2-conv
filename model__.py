import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 残差块
class Residual(nn.Module):

    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1, dropout=False, p=0.2, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.dropout = dropout

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

        if self.dropout:
            Y = self.drop(Y)

        if self.conv3:
            X = self.conv3(X)
        # 要求X和Y具有相同的形状
        # 如果不使用1x1卷积，则两个卷积层的输入（X）和输出（Y）形状相同
        # 使用1x1卷积，可以将输入变换为所需的形状后再做相加的运算
        Y += X
        return F.relu(Y)


class nor_cov(nn.Module):
    """单层卷积网络
    """

    def __init__(self, in_channel, out_channel, normalize=False, dropout=False, p=0.2):
        super(nor_cov, self).__init__()

        self.cov = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

        self.normalize = normalize
        if normalize:
            self.nor = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()
        self.dropout = dropout
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


class dou_cov(nn.Module):
    """双层卷积网络
    """

    def __init__(self, channel, dropout=False, normalize=False, p=0.2):
        super(dou_cov, self).__init__()
        self.cov1 = nor_cov(in_channel=channel, out_channel=channel, dropout=dropout, normalize=normalize, p=p)
        self.cov2 = nor_cov(in_channel=channel, out_channel=channel, dropout=dropout, normalize=normalize, p=p)

    def forward(self, x):
        x = self.cov1(x)
        x = self.cov2(x)
        return x


# 残差模块，由若干个残差块组成
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block = False, dropout=False, p=0.2, normalize=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2, dropout=dropout,
                                  p=p, normalize=normalize))
        else:
            blk.append(Residual(num_channels, num_channels, dropout=dropout,
                                  p=p, normalize=normalize))
    return blk

class CNN_net(nn.Module):
    """自定义卷积网络
    """

    def __init__(self, normalize=False, residual_connection=True, p=0.2, dropout=True):
        super(CNN_net, self).__init__()

        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                        nn.BatchNorm2d(64), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
        

    def forward(self, x):
        for i in range(len(self.fctions)-1):
            x = self.fctions[i](x)
        x = x.reshape(x.shape[0], -1)
        x = self.fctions[-1](x)
        return x


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))

X = torch.rand(size=(1,1,224,224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)