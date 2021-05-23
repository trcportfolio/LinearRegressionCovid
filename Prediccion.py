import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.patches as mpatches
import numpy as np
import datetime as dt
import pandas as pd

dataset = pd.read_csv("TotalesNacionales_T.csv")

y_temp = dataset.iloc[:, 5].values
x_temp = []
for i in range(205):
    x_temp.append(i+1)


X_train = torch.FloatTensor(x_temp)
Y_train = torch.FloatTensor(y_temp)

X_train = X_train.view(205, 1)
Y_train = Y_train.view(205, 1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x


net = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0000001)



net.load_state_dict(torch.load("modelo.pth"))


weight = net.fc1.weight.detach().numpy()
weight = weight[0]
bias = net.fc1.bias.detach().numpy()
bias = bias[0]
 
prediccion_usuario = []
prediccion_usuario_temp = int(input(
    "Ingrese el dia que quiere predecir desde el comienzo del brote(en formato de numero entero):"))
prediccion_usuario.append(prediccion_usuario_temp)
prediccion_usuario = torch.FloatTensor(prediccion_usuario)
prediccion_usuario.view(1, 1)
prediccion_usuario = net(prediccion_usuario).detach().numpy()
prediccion_usuario = prediccion_usuario[0]
print("La cantidad de casos predicha para este dia es:", int(prediccion_usuario))

plt.title('Casos Activos COVID-19 - Chile')
plt.xlabel('Dias desde el comienzo')
plt.ylabel('Cantidad')

plt.plot(X_train, Y_train, "r")
x = np.linspace(0, prediccion_usuario_temp, 1000)
plt.plot(x, weight*x + bias, "b", linestyle="--")

red_patch = mpatches.Patch(color='red', label='Casos Activos')
blue_patch = mpatches.Patch(color='blue', label='Prediccion')
plt.legend(handles=[red_patch, blue_patch])

plt.plot(prediccion_usuario_temp, prediccion_usuario, "go")
plt.show()


