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
from TOOLS.bg_remover import BgRemover


class PAR(nn.Module):

    def __init__(self,extractor,upper_color,lower_color,gender,hat,bag) -> None:
        super(PAR, self).__init__()
        self.extractor = extractor
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
   

class ImageDataset(Dataset):

    classes_name = {'upper_color':1,'lower_color':2,'gender':3,'bag':4,'hat':5}

    def __init__(self, annotations_file, img_dir,class_name, transform=None, target_transform=None,):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels.sample(frac=1)
        self.img_labels = self.img_labels.query(f'{class_name} != -1')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.class_name = class_name
        self.pll = transforms.Compose([transforms.PILToTensor()])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = self.pll(image).to(torch.float32)
        label = self.img_labels.iloc[idx, ImageDataset.classes_name[self.class_name]].astype(np.float32)
        if label[0] != -1:
            label[0] -= 1
        if label[1] != -1:
            label[1] -= 1   
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


class MTImageDataset(Dataset):

    def __init__(self, annotations_file, img_dir,transform=None, target_transform=None,):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels.sample(frac=1)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.pll = transforms.Compose([transforms.PILToTensor()])
        self.data_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=30),  # Losowe obrocenie o maksymalnie 30 stopni
    transforms.RandomHorizontalFlip(p=0.5),  # Lustrzane odbicie z prawdopodobieństwem 0.5
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Zmiana jasności i kontrastu
])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = self.pll(image).to(torch.float32)
        image = self.data_transforms(image)
        label = self.img_labels.iloc[idx,1:6]
        label = torch.tensor(label,dtype= torch.float32)
        if label[0] != -1:
            label[0] -= 1
        if label[1] != -1:
            label[1] -= 1        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    

class CLAHEImageDataset(Dataset):


    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None,):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels.sample(frac=1)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.pll = transforms.Compose([transforms.PILToTensor()])
        self.bgr = BgRemover()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = self.pll(image).to(torch.float32)
        image = self.bgr.clahe(image)
        label = self.img_labels.iloc[idx,1:6]
        label = torch.tensor(label,dtype= torch.float32)
        
        if label[0] != -1:
            label[0] -= 1
        if label[1] != -1:
            label[1] -= 1    
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
    
        return image, label

