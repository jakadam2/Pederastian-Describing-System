import torch.nn as nn
from PAR.cbam import CBAM
import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

class MTPartClassifier(nn.Module):

    def __init__(self,nclasses) -> None:
        super(MTPartClassifier,self).__init__()
        self.attention_module = CBAM(768)
        self.dl1 = nn.Linear(3072,1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dl2 = nn.Linear(1024,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dl3 = nn.Linear(128,64)
        self.dl4 = nn.Linear(64,nclasses)
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
        features = self.dropout(features)
        features = self.relu(features)
        features = self.dl4(features)
        return features


    


class MTPAR(nn.Module):

    def __init__(self,device = 'cuda') -> None:
        super(MTPAR,self).__init__()
        model = models.convnext_small(models.ConvNeXt_Small_Weights.IMAGENET1K_V1).to(device)
        return_node = {'features.7.1' :'stochastic_depth'}
        self.extractor = create_feature_extractor(model, return_node)