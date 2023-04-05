import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.cuda.amp.autocast_mode import autocast

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = False):
        super().__init__()
        self.downsample = downsample
        self.normalpath = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size = 3, stride = 2 if downsample else 1, padding = 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 2 if downsample else 1),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
       return self.normalpath(x) + self.shortcut(x)

class ResNet50(nn.Module):
  def __init__(self, resblock, in_channels, out_features):
    super().__init__() 
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size = 7, stride = 2, padding = 3),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )

    self.conv2 = nn.Sequential(
        nn.MaxPool2d(kernel_size = 3, stride = 2), 
    )  
    self.conv2.add_module("conv2_1", resblock(64, 256))
    for i in range(1, 3):
        self.conv2.add_module(f"conv2_{i + 1}", resblock(256, 256))

    self.conv3 = nn.Sequential()
    self.conv3.add_module("conv3_1", resblock(256, 512, downsample = True))
    for i in range(1, 4):
        self.conv3.add_module(f"conv3_{i + 1}", resblock(512, 512))

    self.conv4 = nn.Sequential()
    self.conv4.add_module("conv4_1", resblock(512, 1024, downsample = True))
    for i in range(1, 6):
        self.conv4.add_module(f"conv4_{i + 1}", resblock(1024, 1024))

    self.conv5 = nn.Sequential()
    self.conv5.add_module("conv5_1", resblock(1024, 2048, downsample = True))
    for i in range(1, 3):
        self.conv5.add_module(f"conv5_{i + 1}", resblock(2048, 2048))

    self.gap = torch.nn.AdaptiveAvgPool2d(1)
    self.fc = torch.nn.Linear(2048, out_features)

  @autocast()
  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)
    out = self.gap(out)
    out = torch.flatten(out, 1)
    out = self.fc(out)
    return out