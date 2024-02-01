import torch.nn as nn
import torch.nn.functional as F
from PAR.cbam import CBAM
import torch
from PAR.resnet_extractor import ResNetExtractor
from torchvision.models.feature_extraction import create_feature_extractor

class Classifier(nn.Module):
    def __init__(self, num_classes=11):
        super(Classifier,self).__init__()
        # self.attention_module = CBAM(768)
        self.attention_module = CBAM(2048)
        # self.attention_module = CBAM(512)
        # self.dl1 = nn.Linear(3072,1024)
        self.dl1 = nn.Linear(8192,1024)
        # self.dl1 = nn.Linear(2048,1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dl2 = nn.Linear(1024,512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dl3 = nn.Linear(512,128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dl4 = nn.Linear(128,64)
        self.dropout = nn.Dropout(0.3)
        self.avg_pool = nn.AvgPool2d((3,3))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.attention_module(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dl1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.dl2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.dl3(x)
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.dl4(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class DMTPAR(nn.Module):

    def __init__(self,device = 'cuda') -> None:
        super(DMTPAR,self).__init__()
        self.extractor = ResNetExtractor().to(device)
        self.upper_color = Classifier(11).to(device)
        self.lower_color = Classifier(11).to(device)
        self.bag_presence = Classifier(1).to(device)
        self.hat_presence = Classifier(1).to(device)
        self.gender = Classifier(1).to(device)
        self.sigmoid = nn.Sigmoid()
        self.dl5_categorical = nn.Linear(64,11).to(device)
        self.dl5_binary = nn.Linear(64,1).to(device)

    def forward(self,x):
        x = self.extractor(x)
        bag_presence = self.sigmoid(self.dl5_binary(self.bag_presence(x)))
        hat_presence = self.sigmoid(self.dl5_binary(self.hat_presence(x)))
        gender = self.sigmoid(self.dl5_binary(self.gender(x)))
        upper_color = self.dl5_categorical(self.upper_color(x))
        lower_color = self.dl5_categorical(self.lower_color(x))
        return upper_color,lower_color,gender,bag_presence ,hat_presence 

class DMTPARpart(nn.Module):

    def __init__(self,model) -> None:
        super(DMTPARpart,self).__init__()
        self._model = create_feature_extractor(model,{'dl5_categorical':'upper_color','dl5_categorical_1':'lower_color'})

    def forward(self,x):
        ans = self._model(x)
        return ans['upper_color'],ans['lower_color']


class DMTLoss(nn.Module):

    def __init__(self) -> None:
        super(DMTLoss,self).__init__()
        self._categorical_loss = nn.CrossEntropyLoss()
        self._binary_loss = nn.BCELoss()
        self._num_classes = 11

    def forward(self,outputs,labels):
        cum_loss = 0

        for j in range(labels.shape[1]):
            mask = labels[:, j] != -1
            if j < 2:
                labels_float = labels[:,j][mask].long()
                cum_loss += self._categorical_loss(outputs[j][mask],labels_float)
            else:
                output = outputs[j][mask].squeeze(1)
                cum_loss += self._binary_loss(output,labels[:,j][mask])

        return cum_loss
    

class PredicitonParser:
    
    color_dict = {0:'black', 1: 'blue',2:'brown',3: 'gray', 4:'green', 5:'orange', 6:'pink', 7:'purple', 8:'red', 9:'white',10: 'yellow'}
    gender_dict = {0:'male',1:'female'}
    bag_dict = {0:False,1:True}
    hat_dict = {0:False,1:True}

    def parse_to_person(self,person,predicts):
        person.upper_color = int(torch.argmax(F.softmax(predicts[0], dim=1)))
        person.lower_color = int(torch.argmax(F.softmax(predicts[1], dim=1)))
        person.gender = int(torch.where(predicts[2] > .5, 1.0, 0.0))
        person.bag = int(torch.where(predicts[3] > .5, 1.0, 0.0))
        person.hat = int(torch.where(predicts[4] > .5, 1.0, 0.0))
        