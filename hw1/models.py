"""Assignment architecture, no BatchNorm specified in the layer table."""
import torch
from torch import nn

class SmallCNN(nn.Sequential):
    def __init__(self):
        layers=[]
        for i,(ci,co,k,s) in enumerate([(3,32,7,2),(32,64,5,1),(64,128,3,2),(128,256,1,1),(256,256,3,2),(256,512,1,1)]):
            layers += [nn.Conv2d(ci,co,k,stride=s,padding=k//2,bias=False), nn.ReLU(inplace=True)]
            if i==0: layers += [nn.MaxPool2d(3,2,1)]
        layers += [nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(512,256),nn.ReLU(inplace=True),nn.Linear(256,100)]
        super().__init__(*layers)
