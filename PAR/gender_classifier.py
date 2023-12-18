import torch
import torch.nn as nn
import torchvision.models as models
from PAR.cbam import CBAM
import torch.nn.functional as F


class GenderClassiefier(nn.Module):
    
    def __init__(self,extractor) -> None:
        super(GenderClassiefier,self).__init__()
        self.extractor = extractor
        self.attention_module = CBAM(768)
        self.dl1 = nn.Linear(3072,128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dl2 = nn.Linear(128,64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dl3 = nn.Linear(64,1)
        self.dropout = nn.Dropout(0.2)
        self.avg_pool = nn.AvgPool2d((3,3))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        for param in self.extractor.parameters():
            param.requires_grad = False

    def forward(self,x):
        features = self.extractor(x)['stochastic_depth']
        features = self.attention_module(features)
        features = self.avg_pool(features)
        features = self.flatten(features)
        features = self.dropout(features)
        features = self.dl1(features)
        features = self.bn1(features)
        features = self.dropout(features)
        features = self.dl2(features)
        features = self.bn2(features)
        features = self.relu(features)
        features = self.dl3(features)
        return F.sigmoid(features)

    def tail(self):
        return nn.Sequential(self.attention_module,self.dl1,self.bn1,self.dropout,self.dl2,self.bn2,self.relu,self.dl3,nn.Sigmoid())
    

class GenderClassiefierBig(nn.Module):
    
    def __init__(self,extractor) -> None:
        super(GenderClassiefierBig,self).__init__()
        self.extractor = extractor
        self.attention_module = CBAM(768)
        self.dl1 = nn.Linear(3072,164)
        self.bn1 = nn.BatchNorm1d(164)
        self.dl2 = nn.Linear(164,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dl3 = nn.Linear(128,64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dl4 = nn.Linear(64,48)
        self.dl5 = nn.Linear(48,1)
        self.dropout = nn.Dropout(0.2)
        self.avg_pool = nn.AvgPool2d((3,3))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        for param in self.extractor.parameters():
            param.requires_grad = False

    def forward(self,x):
        features = self.extractor(x)['stochastic_depth']
        features = self.attention_module(features)
        features = self.avg_pool(features)
        features = self.flatten(features)
        features = self.dropout(features)
        features = self.dl1(features)
        features = self.bn1(features)
        features = self.dropout(features)
        features = self.dl2(features)
        features = self.bn2(features)
        features = self.dropout(features)
        features = self.dl3(features)
        features = self.bn3(features)
        features = self.dropout(features)
        features = self.dl4(features)
        features = self.relu(features)
        features = self.dropout(features)
        features = self.dl5(features)
        return F.sigmoid(features)
