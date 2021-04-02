import torch
x = torch.rand(3,requires_grad = True)
print(x)
y = x+2
print(y)
z = y*y
print(z)
z.sum().backward()
print(x.grad)