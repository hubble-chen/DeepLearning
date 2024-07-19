import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# load the CSV file
train_df = pd.read_csv(r"D:\kaggle\Titanic\train.csv")
test_df = pd.read_csv(r"D:\kaggle\Titanic\test.csv")


# Data Preprocessing
full_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)


# Handling missing values
full_df['Embarked'].fillna(full_df['Embarked'].mode()[0], inplace=True)
full_df['Age'].fillna(full_df['Age'].mean(), inplace=True)
full_df['Fare'].fillna(full_df['Fare'].mean(), inplace=True)
full_df['Cabin'].fillna('NA', inplace=True)


full_df.drop('PassengerId', axis=1, inplace=True)
full_df.drop('Name', axis=1, inplace=True)
full_df.drop(['Ticket', 'Cabin'], axis=1, inplace=True)


# Standardize the values
full_df['Age'] = (full_df['Age'] - full_df['Age'].mean()) / full_df['Age'].std()
full_df['Fare'] = (full_df['Fare'] - full_df['Fare'].mean()) / full_df['Fare'].std()


map = {'male': 1, 'female': 0}
full_df['Sex'] = full_df['Sex'].map(map)
full_data = pd.get_dummies(full_df).astype(float)
train_data = full_data[full_data['Survived'].notnull()]
test_data = full_data[full_data['Survived'].isnull()]

# Create TensorDataset and DataLoader
class TitanicDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = self.data.shape[0]
        self.x_data = torch.from_numpy(self.data.iloc[:, 1:].to_numpy().astype(np.float32))
        self.y_data = torch.from_numpy(self.data[['Survived']].to_numpy().astype(np.float32))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


train_dataset = TitanicDataset(train_data)
dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=32)


# Define the Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 16),
            nn.Sigmoid(),
            nn.Linear(16, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
            x = self.model(x)
            return x


# Instantiate the model
model = Model()


# Loss and Optimizer
criterion = nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# ModelTraining
losses = []
for epoch in range(1000):
    print('epoch:',epoch)
    total_loss = 0
    for data in dataloader:
        inputs, labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("loss:",total_loss.item())
    losses.append(total_loss.detach().numpy())


# loss Visualizing
plt.plot(range(1, 1001), losses)
plt.show()


outputs = (model(train_dataset.x_data) > 0.5).float()
acc = (outputs == train_dataset.y_data.float()).float().sum().item()
acc = acc/train_dataset.len
print(acc)

test_dataset = TitanicDataset(test_data)
y = model(test_dataset.x_data).detach().numpy()
y = (y > 0.5).astype(int)
out = pd.DataFrame({'PassengerId': range(892, 1310), 'Survived': y.flatten()})
out.to_csv("D:\kaggle\Titanic\predict_1.csv",index=False)