'''
Simple CBAM attention module from paper https://arxiv.org/pdf/1807.06521.pdf
During wrtiting this code we follow official github implementation https://github.com/Jongchan/attention-module/tree/master 

'''
import torch
import torch.nn as nn
import torch.nn.functional as F 


class ChannelModule(nn.Module):

    def __init__(self,input,reduction) -> None:
        super(ChannelModule,self).__init__()
        self.dl = nn.Sequential(
            nn.Linear(input,input//reduction),
            nn.ReLU(),
            nn.Linear(input//reduction,input)
        )

    def forward(self,x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        score_max = self.dl(max.view(b,c)).view(b, c, 1, 1)
        score_avg = self.dl(avg.view(b,c)).view(b, c, 1, 1)
        join_score = score_max + score_avg
        return torch.sigmoid(join_score)*x


class SpatialModule(nn.Module):

        def __init__(self) -> None:
            super(SpatialModule,self).__init__()
            self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1)

        def forward(self,x):
            max = torch.max(x,1)[0].unsqueeze(1)
            avg = torch.mean(x,1).unsqueeze(1)
            concat = torch.cat((max,avg), dim=1)
            output = self.conv(concat)
            return torch.sigmoid(output) * x         


class CBAM(nn.Module):

    def __init__(self,channels,reduction = 16) -> None:
        super(CBAM,self).__init__()
        self.channel_module = ChannelModule(channels,reduction)
        self.spatial_module = SpatialModule()

    def forward(self,x):
        attention = self.channel_module(x)
        attention = self.spatial_module(attention)
        return attention + x

