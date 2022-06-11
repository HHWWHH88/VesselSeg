import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Final_code.lib.utils as utils
from torchsummary import summary
# =======================2D-Unet===============================
class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(y, 2, stride=2)

        return x, y


class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_block, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):
        x = self.transconv(x)
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x


class Unet(nn.Module):
    def __init__(self, in_chan=1, out_chan=1):
        self.in_chan = in_chan
        self.out_chan = out_chan
        super(Unet, self).__init__()
        self.down1 = Downsample_block(in_chan, 64)
        self.down2 = Downsample_block(64, 128)
        self.down3 = Downsample_block(128, 256)
        self.down4 = Downsample_block(256, 512)
        self.conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.up4 = Upsample_block(1024, 512)
        self.up3 = Upsample_block(512, 256)
        self.up2 = Upsample_block(256, 128)
        self.up1 = Upsample_block(128, 64)
        self.outconv = nn.Conv2d(64, out_chan, 1)
        self.outconvp1 = nn.Conv2d(64, out_chan, 1)
        self.outconvm1 = nn.Conv2d(64, out_chan, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        x = F.dropout2d(F.relu(self.bn1(self.conv1(x))))
        x = F.dropout2d(F.relu(self.bn2(self.conv2(x))))
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x1 = self.outconv(x)
        x1 = self.sigmoid(x1)
        return x1
# ========================IPN===============================

class IPN(nn.Module):
    def __init__(self, in_channels, PLM_NUM=5, filter_size=3, LAYER_NUM=3, NUM_OF_CLASS=2, pooling_size=[], input_size=[4, 1, 160, 100, 100]):
        super(IPN, self).__init__()
        self.pooling_size = pooling_size
        self.input_size = input_size
        self.features = np.ones(PLM_NUM, dtype='int32') * 64
        if not self.pooling_size:
            self.combine = utils.cal_downsampling_size_combine(self.input_size[2], PLM_NUM)
            self.pooling_size = self.combine
        else:
            PLM_NUM = len(pooling_size)
        self.NUM_OF_CLASS = NUM_OF_CLASS
        self.in_channels = in_channels
        self.filter_size = filter_size
        self.print_aras()
        self.LAYER_NUM = LAYER_NUM

        self.layer1 = self._make_layer(channel=self.in_channels, features_cnt=0, layer_num=self.LAYER_NUM,
                                       pooling_cnt=0)# output shape (3, 64, 32, 100, 100)
        self.layer2 = self._make_layer(channel=self.features[0], features_cnt=1, layer_num=self.LAYER_NUM,
                                       pooling_cnt=1)# output shape (3, 64, 8, 100, 100)
        self.layer3 = self._make_layer(channel=self.features[1], features_cnt=2, layer_num=self.LAYER_NUM,
                                       pooling_cnt=2)# output shape (3, 64, 4, 100, 100)
        self.layer4 = self._make_layer(channel=self.features[2], features_cnt=3, layer_num=self.LAYER_NUM,
                                       pooling_cnt=3)# output shape (3, 64, 2, 100, 100)
        self.layer5 = self._make_layer(channel=self.features[3], features_cnt=4, layer_num=self.LAYER_NUM,
                                       pooling_cnt=4)# output shape (3, 64, 1, 100, 100)
        #output
        self.layer_op = nn.Sequential(nn.Conv3d(in_channels=64, out_channels=self.NUM_OF_CLASS, kernel_size=filter_size, stride=1, padding=1), nn.ReLU())

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer_op(x)
        softmax = nn.Softmax(dim=-1)
        sf = softmax(x)
        #pred = torch.argmax(sf, dim=1)
        return sf

    def _make_layer(self, channel, features_cnt, layer_num, pooling_cnt):
        layers = []
        layers.append(
            nn.Conv3d(in_channels=channel, out_channels=self.features[features_cnt], kernel_size=self.filter_size,
                      stride=1,
                      padding=1))
        layers.append(nn.ReLU())
        for _ in range(1,layer_num):
            layers.append(
                nn.Conv3d(in_channels=self.features[features_cnt], out_channels=self.features[features_cnt],
                          kernel_size=self.filter_size,
                          stride=1,
                          padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool3d(kernel_size=[self.pooling_size[pooling_cnt], 1, 1],
                                   stride=[self.pooling_size[pooling_cnt], 1, 1]))
        return nn.Sequential(*layers)

    def print_aras(self):
        print('')
        print('-----------------  model paras ------------------')
        resize = 1
        print('PLM DS SIZE: ', end='')
        for index in self.pooling_size:
            resize *= index
        print('{}->{} = '.format(self.input_size[2], resize), end='')
        for i, s in enumerate(self.pooling_size):
            if i == 0:
                print(str(s), end='')
            else:
                print('x' + str(s), end='')

        print('')
        print('conv channel nums : ', end='')
        for f in self.features:
            print(f, ',', end='')
        print('')
        print('---------------------  end ----------------------')
        print('')

# summary(IPN(in_channels=1),(1, 160, 100, 100))
# summary(Unet(),(1,64,64))