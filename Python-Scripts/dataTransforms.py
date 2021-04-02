import torch 
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np 
import math

class WineDataset(Dataset):
    def __init__(self,transform=None):
        data = np.loadtxt('C:\\Users\\Swagata Misra\\Documents\\GitHub\\PyTorchBasics\\Python-Scripts\\Data\\wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.x = (data[:,1:])
        self.y = (data[:,[0]])
        self.n_samples = data.shape[0]
        self.transform = transform
    def __getitem__(self,index):
        sample = self.x[index],self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self,sample):
        inp,tar = sample
        return torch.from_numpy(inp), torch.from_numpy(tar)

class MulTransform:
    def __init__(self,factor):
        self.factor=factor
    def __call__(self,sample):
        inp,tar = sample
        inp*=self.factor
        return inp,tar


composed = torchvision.transforms.Compose([ToTensor(),MulTransform(2)])
d = WineDataset()
print(f'Without mul {d[0]}')
dataset = WineDataset(transform=composed)
print(f'With mul {dataset[0]}')


