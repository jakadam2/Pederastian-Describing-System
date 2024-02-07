import torch.nn as nn
from PAR.cbam import CBAM
import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from PAR.resnet_extractor import ResNetExtractor


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


class AMTLoss(nn.Module):

    steps = [11,11,2,2,2]

    def __init__(self) -> None:
        super(AMTLoss,self).__init__()
        self._loss = nn.CrossEntropyLoss() 

    def forward(self,predicts,labels):
        cum_loss = 0
        j = 0
        labels = labels.squeeze(1)

        for i in range(labels.shape[1]):
            mask = labels[:,i] != -1
            cum_loss += self._loss(predicts[:,j:j + AMTLoss.steps[i] + 1][mask],labels[:,i][mask])
            j += AMTLoss.steps[i]
        return cum_loss
    

class Classifier(nn.Module):

    def __init__(self, num_classes=11):
        super(Classifier,self).__init__()
        self.attention_module = CBAM(512)
        self.dl1 = nn.Linear(2048,128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dl4 = nn.Linear(128,64)
        self.avg_pool = nn.AvgPool2d((3,3))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.attention_module(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dl1(x)

        x = self.bn3(x)
        x = self.dl4(x)
        x = self.relu(x)
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