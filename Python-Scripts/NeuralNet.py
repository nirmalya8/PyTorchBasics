import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

input_size = 784
hidden_size = 500
epochs = 5
num_classes=10
batch_size = 100
lr = 0.001

train = torchvision.datasets.MNIST(root='./Data', train=True, transform=transforms.ToTensor(), download=True)
test = torchvision.datasets.MNIST(root='./Data', train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset = train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test,batch_size=batch_size,shuffle=True)

def plotimgs(samples):
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(samples[i][0],cmap='gray')
    plt.show()

examples=iter(train_loader)
samples,labels = examples.next()
#plotimgs(samples)

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

model = NeuralNet(input_size,hidden_size,num_classes)

loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr=lr)

for epoch in range(epochs):
    for i,(images,labels) in enumerate(train_loader):
        images = images.reshape(-1,784)
        
        yhat = model(images)
        l = loss(yhat,labels)

        l.backward()
        optim.step()

        optim.zero_grad()
        if (i+1)%100 == 0:
            print(f'Loss at epoch {epoch+1} step {i+1} is {l}')

    if (epoch+1)%1==0:
        print(f"Loss at epoch {epoch+1} = {l}")


with torch.no_grad():
    corr,tot = 0,0
    for images, labels in test_loader:
        images = images.reshape(-1,784)
        out  = model(images)

        _,pred = torch.max(out,1)
        tot+=labels.shape[0]
        corr+=(pred==labels).sum().item()
    acc = 100*corr/tot
    print(f'Accuracy = {acc}')