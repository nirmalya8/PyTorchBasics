import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt 

#data preparation

x_np,y_np = datasets.make_regression(n_samples=1000,n_features=1,noise=20,random_state=1)
x = torch.from_numpy(x_np.astype(np.float32))
y = torch.from_numpy(y_np.astype(np.float32))
y = y.view(y.shape[0],1)

n_samples,n_features = x.shape

inp_size = n_features
out_size = 1
model = nn.Linear(inp_size,out_size)

lr = 0.01
epochs = 10000
loss = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(),lr=lr)

for epoch in range(epochs):
    yhat = model(x)
    l = loss(y,yhat)
    l.backward()
    optim.step()
    optim.zero_grad()
    if (epoch+1)%100==0:
        print(f"Loss at epoch {epoch+1} = {l}")

pred = model(x).detach()
plt.plot(x_np,y_np,'r+')
plt.plot(x_np,pred,'b')
plt.show()
