import torch 
import torch.nn as nn
x = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)
n_samples, n_features = x.shape
print(f'#samples: {n_samples}, #features: {n_features}')
input_size = n_features
output_size = n_features
X_test = torch.tensor([20], dtype=torch.float32)

model = nn.Linear(input_size,output_size)

lr = 0.01
epochs = 1000
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr)
#training
for epoch in range(epochs):
    yhat = model(x)
    l = loss(y,yhat)
    print(f"Loss at epoch {epoch+1} = {l}")
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
print(f'Prediction for {X_test} is {model(X_test)}')