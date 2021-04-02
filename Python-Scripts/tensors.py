import torch
a = torch.empty(5,3) #Enter the size separated by comas
print(a)

zero = torch.zeros(5,3,dtype=float)
print(zero)


one = torch.ones(5,3,dtype=float)
print(one)

t = torch.tensor([1,2,3.])
print(t)

one = one.reshape(3,5)
print(one)