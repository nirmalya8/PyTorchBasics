import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as f

epochs = 20
batch_size = 4
lr = 0.001

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train = torchvision.datasets.CIFAR10(root='./DataCIFAR', train=True,  download=True,transform=transform)
test = torchvision.datasets.CIFAR10(root='./DataCIFAR', train=False,  download=True,transform=transform)

train_loader = torch.utils.data.DataLoader(dataset = train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test,batch_size=batch_size,shuffle=True)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1  = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    
    def forward(self,x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x  


model = ConvNet()

loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(),lr=lr)

for epoch in range(epochs):
    for i,(images,labels) in enumerate(train_loader):
        output = model(images)
        l = loss(output,labels)
        optim.zero_grad()

        l.backward()
        optim.step()

        if (i+1)%100 == 0:
            print(f'Loss at epoch {epoch+1} step {i+1} is {l}')

    if (epoch+1)%1==0:
        print(f"Loss at epoch {epoch+1} = {l}")


with torch.no_grad():
    corr =0
    tot = 0 
    class_corr = [0 for i in range(10)]
    class_tot = [0 for i in range(10)]
    for images, labels in test_loader:
        out = model(images)
        _,pred = torch.max(out,1)
        tot+=labels.shape[0]
        corr+=(pred == labels).sum().item()
        
        for i in range(batch_size):
            l = labels[i]
            p = pred[i]
            if(l==p):
                class_corr[l]+=1 
            class_tot[l]+=1
    acc = 100.0*corr/tot 
    print(f'Total accuracy = {acc}')

    for i in range(10):
        a = 100.0*class_corr[i]/class_tot[i]
        print(f'Accuracy of {classes[i]} is {a}')