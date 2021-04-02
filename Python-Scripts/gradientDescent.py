import torch 
x = torch.tensor([1,2,3,4],dtype=float)
y = torch.tensor([2,4,6,8],dtype=float)
w = torch.tensor(0.0,dtype=float,requires_grad=True)

def forward(x):
    return w*x 

def loss(y,yhat):
    return ((yhat-y)**2).mean()

lr = 0.01
epochs = 1000
#training
for epoch in range(epochs):
    yhat = forward(x)
    l = loss(y,yhat)
    print(f"Loss at epoch {epoch+1} = {l}")
    l.backward()
    with torch.no_grad():
        w1 = w - lr*w.grad
        w.copy_(w1)
        w.grad.zero_()
print(f"Final w = {w}")