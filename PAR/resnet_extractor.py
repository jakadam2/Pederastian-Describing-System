import torch
import torch.nn as nn
import torchvision.models as models

class ResNetExtractor(nn.Module):
    '''
    INPUT SHAPE = (batch_size, 3, 224, 224)
    OUTPUT SHAPE = (batch_size, 2048, 7, 7)
    '''
    def __init__(self,device = 'cuda') -> None:
        super(ResNetExtractor,self).__init__()
        model = models.resnet50(pretrained=True).to(device)
<<<<<<< HEAD
        # model = models.resnet34(pretrained=True).to(device)
=======
        #model = models.resnet34(pretrained=True).to(device)
>>>>>>> e5d5623 (time fix)
        self.extractor = torch.nn.Sequential(*(list(model.children())[:-2]))  


    def forward(self,x) -> None:
        return self.extractor(x)