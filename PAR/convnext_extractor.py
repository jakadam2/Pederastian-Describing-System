import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

class ConvexNextExtractor(nn.Module):
    # INPUT SHAPE = (batch_size, 3, 224, 224)
    # OUTPUT SHAPE = (batch_size, 768, 7, 7)
    def __init__(self,device = 'cuda') -> None:
        super(ConvexNextExtractor,self).__init__()
        model = models.convnext_small(models.ConvNeXt_Small_Weights.IMAGENET1K_V1).to(device)
        return_node = {'features.7.1' :'stochastic_depth'}
        self.extractor = create_feature_extractor(model, return_node)

    def forward(self,x) -> None:
        return self.extractor(x)