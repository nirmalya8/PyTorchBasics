import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

bc = datasets.load_breast_cancer()
x,y = bc.data,bc.target

n_samples,n_features = x.shape
print(n_samples,n_features)
print(x[0],y[0])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.model = nn.Linear(n_input_features,1)
    
    def forward(self,x):
        ypred = torch.sigmoid(self.model(x))
        return ypred

model = LogisticRegression(n_features)

lr=0.01
epochs = 1000
loss = nn.BCELoss()
#optim = torch.optim.SGD(model.parameters(),lr=lr)
optim = torch.optim.Adam(model.parameters(),lr=lr)
for epoch in range(epochs):
    ypred = model(x_train)
    l = loss(ypred,y_train)
    l.backward()
    optim.step()
    optim.zero_grad()
    if (epoch+1)%100==0:
        print(f"Loss at epoch {epoch+1} = {l}")
with torch.no_grad():
    testpred = model(x_test)
    l = loss(y_test,testpred)
    testpredclass = testpred.round()
    acc = testpredclass.eq(y_test).sum()/float(y_test.shape[0])
    cm = confusion_matrix(y_test,testpredclass)
    print(f'Loss in test set BCE: {l}' )
    print(f'Accuracy : {acc}')
    print(f"Confusion Matrix: {cm}")