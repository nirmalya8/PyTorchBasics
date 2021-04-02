import torch 
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np 
import math

class WineDataset(Dataset):
    def __init__(self):
        data = np.loadtxt('C:\\Users\\Swagata Misra\\Documents\\GitHub\\PyTorchBasics\\Python-Scripts\\Data\\wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.x = torch.from_numpy(data[:,1:])
        self.y = torch.from_numpy(data[:,[0]])
        self.n_samples = data.shape[0]
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.n_samples

dataset = WineDataset()

data = DataLoader(dataset=dataset,batch_size=4,shuffle=True)

epochs = 1000
total_samples = len(dataset)
n_iter = math.ceil(total_samples/4) 

dataiter = iter(data)
data = dataiter.next()
inputs, targets = data
print(inputs.shape, targets.shape)
