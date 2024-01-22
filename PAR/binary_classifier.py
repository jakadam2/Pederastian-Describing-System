import torch.nn as nn
from PAR.cbam import CBAM
import torch

class BinaryClassiefier(nn.Module):
    
    def __init__(self) -> None:
        super(BinaryClassiefier,self).__init__()
        self.attention_module = CBAM(768)
        self.dl1 = nn.Linear(3072,1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dl2 = nn.Linear(1024,512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dl3 = nn.Linear(512,128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dl4 = nn.Linear(128,64)
        self.dl5 = nn.Linear(64,1)
        self.dropout = nn.Dropout(0.3)
        self.avg_pool = nn.AvgPool2d((3,3))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self,x):
        features = self.attention_module(x)
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
        features = torch.sigmoid(features)
        return features