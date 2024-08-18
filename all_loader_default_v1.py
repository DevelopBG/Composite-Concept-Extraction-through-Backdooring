import torch
import cv2
import csv
import os
import pandas as pd
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import transforms as tr, datasets as ds
from torch.utils.data.dataloader import DataLoader
# from sampler import ContinuousBatchSampler

import numpy as np
from PIL import Image
import wget
from zipfile import ZipFile
random_seed = 1234 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

def get_transform(dataset, train=True):
    if dataset == "mnist":
        h = w = 28
    elif dataset in ("cifar10", "gtsrb"):
        h = w = 32
    elif dataset in ("ytf"):
        h = 55
        w = 47
    else:
        raise ValueError("Do not support dataset={args.dataset}!")

    transforms = list()
    transforms.append(tr.Resize((h, w)))

    if train:
        transforms.append(tr.RandomCrop((h, w), padding=2))
        if dataset != "mnist":
            transforms.append(tr.RandomRotation(10))

        if dataset == "cifar10":
            transforms.append(tr.RandomHorizontalFlip(p=0.5))
    transforms.append(tr.ToTensor())
    return tr.Compose(transforms)

def get_dataset_manager(root_dir,dataset = 'cifar10',batch_size = 64, shuffle =True,num_workers=0 ):
    """
    dataset = 'gtsrb','cifar10','svhn','timagenet'
    """

    train_transform = get_transform(dataset, train=True)
    test_transform = get_transform(dataset, train=False)

    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=root_dir, download=True, transform=train_transform, train=True)
        test_dataset = datasets.CIFAR10(root=root_dir, download=True, transform=test_transform, train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # train_sampler = ContinuousBatchSampler(len(train_dataset), num_repeats=5, batch_size=batch_size, shuffle=True)

       ### DATA LOADERS
        # train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    if dataset == 'svhn':
        train_dataset = datasets.SVHN(root=root_dir, download=True, transform=train_transform, split = 'train')
        test_dataset = datasets.SVHN(root=root_dir, download=True, transform=test_transform, split = 'test')
    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_dataset,test_dataset,train_loader,test_loader
    

    
 