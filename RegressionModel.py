import pandas as pd
import numpy as np
import datetime as dt
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


dataset = pd.read_csv("TotalesNacionales_T.csv")

y_temp = dataset.iloc[:, 5].values
x_temp = []
for i in range(205):
    x_temp.append(i+1)


X_train = torch.FloatTensor(x_temp)
Y_train = torch.FloatTensor(y_temp)

X_train = X_train.view(205, 1)
Y_train = Y_train.view(205, 1)

print(X_train)


model = nn.Linear(1, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001)

for epoch in range(1000):

    y_pred = model(X_train)
    loss = criterion(y_pred, Y_train)
    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % 1 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


torch.save(model.state_dict(), "modelo2.pth")

predicted = model(X_train).detach().numpy()
plt.plot(X_train, Y_train, "r")
plt.plot(X_train, predicted, "b")
plt.show()


