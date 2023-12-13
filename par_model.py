import cv2 as cv
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
import pandas as pd
from PIL import Image
import numpy as np

class PAR(nn.Module):

    def __init__(self,upper_color,lower_color,gender,hat,bag) -> None:
        super(PAR, self).__init__()
        self.extractor = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.extractor.eval()
        self.extractor = nn.Sequential(*list(self.extractor.children())[:-1])

        self.upper_color = upper_color
        self.lower_color = lower_color
        self.gender = gender
        self.hat = hat
        self.bag = bag

    def forward(self,x):
        features = self.extractor(x)
        upper_color = self.upper_color(features)
        lower_color = self.lower_color(features)
        hat = self.hat(features)
        bag = self.bag(features)
        gender = self.gender(features)
        return torch.stack((upper_color,lower_color,gender,hat,bag))


class BinaryMobilnetClassifier(nn.Module):

    def __init__(self,extractor) -> None:
        super(BinaryMobilnetClassifier,self).__init__()
        self.extractor = extractor
        for param in self.extractor.parameters():
            param.requires_grad = False
        self.nl = nn.Sequential(nn.AvgPool2d(kernel_size = (7,7)),
                                nn.Flatten(1),
                                nn.Dropout(0.2),
                                nn.Linear(in_features=1280,out_features=1),
                                nn.Sigmoid())

    def forward(self,x):
        features = self.extractor(x)
        result = self.nl(features)
        return result
    

class MultiMobilnetClassifier(nn.Module):

    def __init__(self,extractor,n_labels) -> None:
        super(MultiMobilnetClassifier,self).__init__()
        self.extractor = extractor
        for param in self.extractor.parameters():
            param.requires_grad = False
        self.nl = nn.Sequential(nn.Linear(1280,n_labels),nn.Softmax(1))

    def forward(self,x):
        features = self.extractor(x)
        
        return self.nl(features)
    

class ImageDataset(Dataset):

    classes_name = {'upper_color':1,'lower_color':2,'gender':3,'bag':4,'hat':5}

    def __init__(self, annotations_file, img_dir,class_name, transform=None, target_transform=None,):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels.query(f'{class_name} != -1')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.class_name = class_name

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).to(torch.float32)
        label = self.img_labels.iloc[idx, ImageDataset.classes_name[self.class_name]].astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    