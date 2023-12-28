import torch
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor


class Resnet50Extractor(torch.nn.Module):
    def __init__(self):
        super(Resnet50Extractor, self).__init__()
        m = resnet50(pretrained=True)
        self.body = create_feature_extractor(
            m, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([4])})

    def forward(self, x):
        x = self.body(x)
        return x
