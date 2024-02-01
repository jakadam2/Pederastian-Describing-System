import torch.nn as nn
from PAR.cbam import CBAM
import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

class AMTPartClassifier(nn.Module):

    def __init__(self,nclasses) -> None:
        super(AMTPartClassifier,self).__init__()
        self.attention_module = CBAM(512)
        self.dl1 = nn.Linear(2048,128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dl2 = nn.Linear(128,nclasses)
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
        return features
    

class AMTPAR(nn.Module):

    def __init__(self,device = 'cuda') -> None:
        super(AMTPAR,self).__init__()
        return_node = {'layer4.1':'relu_1'}
        model = models.resnet18(models.ResNet18_Weights.IMAGENET1K_V1).to(device)
        self.extractor = create_feature_extractor(model, return_node)
        self.extractor = create_feature_extractor(model, return_node)
        self.upper_color = AMTPartClassifier(11).to(device)
        self.lower_color = AMTPartClassifier(11).to(device)
        self.bag = AMTPartClassifier(2).to(device)
        self.hat = AMTPartClassifier(2).to(device)
        self.gender = AMTPartClassifier(2).to(device)

    def forward(self,x):
        features = self.extractor(x)['relu_1']
        bag = self.bag(features)
        hat = self.hat(features)
        gender = self.gender(features)
        upper_color = self.upper_color(features)
        lower_color = self.lower_color(features)
        return torch.hstack((upper_color,lower_color,gender,hat,bag))


class AMTPARpart(nn.Module):

    def __init__(self,model) -> None:
        super(AMTPARpart,self).__init__()
        self._model = create_feature_extractor(model,{'hat.dl2':'hat','gender.dl2':'gender','bag.dl2':'bag'})

    def forward(self,x):
        ans = self._model(x)
        return ans['bag'],ans['gender'],ans['hat']



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
        return cum_loss
    

class PredicitonParser:
    
    color_dict = {0:'black', 1: 'blue',2:'brown',3: 'gray', 4:'green', 5:'orange', 6:'pink', 7:'purple', 8:'red', 9:'white',10: 'yellow'}
    gender_dict = {0:'male',1:'female'}
    bag_dict = {0:False,1:True}
    hat_dict = {0:False,1:True}

    def parse_to_person(self,person,predicts):
        person.upper_color,person.lower_color,person.gender,person.hat,person.bag = self.parse_prediction(predicts)

    @classmethod
    def parse_prediction(cls,predicts):
        predicts = predicts.squeeze(0)
        upper_color = predicts[0:11]
        lower_color = predicts[11:22]
        gender = predicts[22:24]
        hat = predicts[24:26]
        bag = predicts[26:28]
        return cls.color_dict[int(torch.argmax(upper_color))],cls.color_dict[int(torch.argmax(lower_color))],cls.gender_dict[int(torch.argmax(gender))],cls.hat_dict[int(torch.argmax(hat))],cls.bag_dict[int(torch.argmax(bag))]


    