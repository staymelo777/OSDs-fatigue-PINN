import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('dataset.csv')

X = data.iloc[:, [0, 1, 2]].values  # β, S, R
y = data.iloc[:, 4].values  # logN

β, S, R = X[:, 0], X[:, 1], X[:, 2]
α = S * (2 / (1 - R)) ** 0.2226
computed_logN = np.log(α) * β + S
enhanced_inputs = np.concatenate((X, computed_logN.reshape(-1,1)), axis=1)

#
x_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))
X = x_scaler.fit_transform(enhanced_inputs)
Y = y_scaler.fit_transform(y.reshape(-1,1))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)
        self.fc4 = nn.Linear(4, 1)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logN = self.fc4(x)
        return logN

net = CustomNetwork()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

import matplotlib.pyplot as plt

weight1 = 0.97
weight2 = 0.03
losses = []

predictions, actuals = [], []

best_loss = float('inf')
best_model_state_dict = None
net.train()
for epoch in range(1000):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # print(inputs)
        # β, S, R = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        # α = S * (2 / (1 - R)) ** 0.5
        # computed_logN = torch.log(α) * β + S
        # print(computed_logN)
        # enhanced_inputs = torch.cat((inputs, computed_logN.unsqueeze(1)), dim=1)
        output_logN = net(inputs)
        computed_log = inputs[:,-1]
        loss1 = criterion(output_logN, labels.view(-1, 1))
        loss2 = criterion(output_logN, computed_log.reshape(-1,1))
        loss = weight1 * loss1 + weight2 * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)
    print(f'[Epoch {epoch + 1}] loss: {epoch_loss:.3f}')

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_state_dict = net.state_dict()

torch.save(best_model_state_dict, 'best_model.pth')
net.eval()
net.load_state_dict(torch.load('best_model.pth'))
predictions = []
for data in test_loader:
    inputs, labels = data
    # β, S, R = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    # α = S * (2 / (1 - R)) ** 0.5
    # computed_logN = torch.log(α) * β + S
    # enhanced_inputs = torch.cat((inputs, computed_logN.unsqueeze(1)), dim=1)
    output_logN = net(inputs)
    predictions.extend(output_logN.view(-1).cpu().detach().numpy())
#
# for epoch in range(1000):
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         β, S, R = inputs[:, 0], inputs[:, 1], inputs[:, 2]
#         α = S * (2 / (1 - R)) ** 0.5
#         computed_logN = torch.log(α) * β + S
#         enhanced_inputs = torch.cat((inputs, computed_logN.unsqueeze(1)), dim=1)
#         output_logN = net(enhanced_inputs)
#         print(computed_logN, output_logN)
#         loss1 = criterion(output_logN, labels.view(-1, 1))
#         loss2 = criterion(output_logN, computed_logN.unsqueeze(1))
#         # print(loss1, loss2)
#         loss = weight1 * loss1 + weight2 * loss2
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#
#         if epoch == 999:  # 在最后一个epoch收集预测和实际值
#             predictions.extend(output_logN.view(-1).cpu().detach().numpy())
#             actuals.extend(labels.view(-1).cpu().detach().numpy())
#
#     epoch_loss = running_loss / len(train_loader)
#     losses.append(epoch_loss)
#     print(f'[Epoch {epoch + 1}] loss: {epoch_loss:.3f}')
predictions = y_scaler.inverse_transform(np.array(predictions).reshape(-1,1))
y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1))


plt.figure(figsize=(10, 6))
plt.plot(range(1, 1001), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Values', marker='o')
plt.plot(predictions, label='Predicted Values', marker='x')
plt.title('Comparison of Predicted and Actual Values')
plt.xlabel('Sample Index')
plt.ylabel('logN')
plt.legend()
plt.grid(True)
plt.show()
