import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from utils import trigger_2


class MyDataset(Dataset):
    def __init__(self, data, target,train=True):
        self.data = data
        self.target = target
        if train:
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32,32)),
            transforms.RandomCrop((32,32), padding=2),
            # transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            ])            
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
       
        y = self.target[index]

        return x, y
    
class MyDataset_v1(Dataset):
    def __init__(self, data, target,train=True):
        self.data = data
        self.target = target
        if train:
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32,32)),
            transforms.RandomCrop((32,32), padding=2),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            ])            
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
       
        y = self.target[index]

        return x, y
    
