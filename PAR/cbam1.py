import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAtention(nn.Module):

    def __init__(self,in_features) -> None:
        super(ChannelAtention,self).__init__()
        self.dl1 = nn.Linear(in_features,in_features//16)
        self.dl2 = nn.Linear(in_features//16,in_features)
        self.bn = nn.BatchNorm1d(in_features//16)

    def _attention(self,x) -> torch.Tensor:
        x = nn.Flatten(0)
        x = self.dl1(x)
        x = self.bn(x)
        return self.dl2(x)

    def forward(self,x) -> torch.Tensor:
        avg_pool_features = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool_features= F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avg_pool_score = self._attention(avg_pool_features)
        max_pool_score = self._attention(max_pool_features)
        join_score = max_pool_score + avg_pool_score
        return F.sigmoid(join_score)
    
class SpatialAtention(nn.Module):

    def __init__(self,in_features) -> None:
        super(SpatialAtention,self).__init__()
        self.conv = nn.Conv2d(in_features,in_features,(7,7))

    def forward(self,x) -> torch.Tensor:
        avg_pool_features = F.avg_pool2d(x,x.size(1),x.size(1))
        max_pool_features = F.avg_pool2d(x,x.size(1),x.size(1))
        join_features = torch.cat((max_pool_features,avg_pool_features),2)
        return self.conv(join_features)
    
class CBAM(nn.Module):

    def __init__(self,shape) -> None:
        super(CBAM,self).__init__()
        self.channel_module = ChannelAtention(shape[2])
        self.spatial_module = SpatialAtention(shape[0:2])

    def forward(self,x) -> torch.Tensor:
        chanel_weights = 1 + self.channel_module(x)
        x *= chanel_weights
        spatial_weights = 1 + self.spatial_module(x)
        x *= spatial_weights
        return x
