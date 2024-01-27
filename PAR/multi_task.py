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
        model = models.convnext_tiny(models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device)
        return_node = {'features.7.1' :'add'}
        self.extractor = create_feature_extractor(model, return_node)
        self.upper_color = MTPartClassifier(11).to(device)
        self.lower_color = MTPartClassifier(11).to(device)
        self.bag = MTPartClassifier(2).to(device)
        self.hat = MTPartClassifier(2).to(device)
        self.gender = MTPartClassifier(2).to(device)

    def forward(self,x):
        features = self.extractor(x)['add']
        bag = self.bag(features)
        hat = self.hat(features)
        gender = self.gender(features)
        upper_color = self.upper_color(features)
        lower_color = self.lower_color(features)
        return torch.hstack((upper_color,lower_color,gender,hat,bag))
    

class MTLoss(nn.Module):

    steps = [11,11,2,2,2]

    def __init__(self) -> None:
        super(MTLoss,self).__init__()
        self._loss = nn.CrossEntropyLoss() 

    def forward(self,predicts,labels):
        cum_loss = 0
        j = 0
        labels = labels.squeeze(1)

        for i in range(labels.shape[1]):
            mask = labels[:,i] != -1
            cum_loss += self._loss(predicts[:,j:j + MTLoss.steps[i] + 1][mask],labels[:,i][mask])
            j += MTLoss.steps[i]
        print(cum_loss)
        return cum_loss

    