from PAR.cbam import CBAM
import torchvision.models as models
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
import torch


class MobilnetClassifier(nn.Module):

    def __init__(self) -> None:
        super(MobilnetClassifier,self).__init__()
        self.mobilnet = create_feature_extractor(models.mobilenet_v3_small(models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),{'features.12':'0'})
        self.attention_module2 = CBAM(576)
        self.avg_pool = nn.AvgPool2d((2,2))
        self.flatten = nn.Flatten()
        self.ll1 = nn.Linear(5184,1280)
        self.dropout = nn.Dropout(p = 0.35)
        self.ll2 = nn.Linear(1280,512)
        self.ll3 = nn.Linear(512,1)
        pass

    def forward(self,x):
        x = self.mobilnet(x)['0']
        x = self.attention_module2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.ll1(x)
        x = self.dropout(x)
        x = self.ll2(x)
        x = self.ll3(x)
        return torch.sigmoid(x)