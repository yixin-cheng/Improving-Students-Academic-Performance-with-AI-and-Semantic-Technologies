# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from preprocessor import Preprocessor
from feature_selector import Selector
from sklearn.model_selection import KFold
import random
import torch.optim as optim
from torch.nn import functional as F
import torchvision
from torchvision import datasets,transforms
import torchvision.transforms as transforms
initial_path='data/historicosFinal.csv'
target_path= 'data/preprocessing.csv'
ga_path= 'data/ga_data.csv'

Preprocessor('data/historicosFinal.csv', target_path)
Selector(target_path,ga_path) # apply GA for feature selection

if torch.cuda.is_available():
    device = torch.device("cuda:2")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Hyper Parameters
input_size = len(pd.read_csv(ga_path).columns)-1
hidden_size = 50
num_classes = 2
num_epochs = 500
batch_size = 10
learning_rate = 0.01


# define a function to plot confusion matrix
def plot_confusion(input_sample, num_classes, des_output, actual_output):
    confusion = torch.zeros(num_classes, num_classes)
    for i in range(input_sample):
        actual_class = actual_output[i]
        predicted_class = des_output[i]

        confusion[actual_class][predicted_class] += 1

    return confusion


"""
Step 1: Load data and pre-process data
Here we use data loader to read data
"""


# define a customise torch dataset
class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data_tensor = torch.Tensor(df.values)

    # a function to get items by index
    def __getitem__(self, index):
        obj = self.data_tensor[index]
        input = self.data_tensor[index][0:-1]
        target = self.data_tensor[index][-1] - 1

        return input, target

    # a function to count samples
    def __len__(self):
        n, _ = self.data_tensor.shape
        return n

# load all data
data = pd.read_csv('glass.csv', header=None, index_col=0)

# normalise input data
for column in data.columns[:-1]:
    # the last column is target
    data[column] = data.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

# randomly split data into training set (80%) and testing set (20%)
msk = np.random.rand(len(data)) < 0.8
train_data = data[msk]
test_data = data[~msk]

# define train dataset and a data loader
train_dataset = DataFrameDataset(df=train_data)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


"""
Step 2: Define a neural network 

Here we build a neural network with one hidden layer.
    input layer: n neurons, representing the features of status of students
    hidden layer: 50 neurons, using Sigmoid/Tanh/Tanh/Softmax as activation function
    
    output layer: 2 neurons, representing the type of dropouts
"""


# Neural Network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.Tanh=nn.Tanh()
        self.Softmax=nn.Softmax()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out=self.Tanh(out)
        out=self.Tanh(out)
        out = self.Softmax(out)
        out = self.fc2(out)
        return out


net = Net(input_size, hidden_size, num_classes).to(device=device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# store all losses for visualisation
all_losses = []

# train the model by batch
for epoch in range(num_epochs):
    total = 0
    correct = 0
    total_loss = 0
    for step, (batch_x, batch_y) in enumerate(train_loader):
        X = batch_x
        Y = batch_y.long()
        # print(X)
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(X)
        loss = criterion(outputs, Y)
        all_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if (epoch % 50 == 0):
            _, predicted = torch.max(outputs, 1)
            # calculate and print accuracy
            total = total + predicted.size(0)
            correct = correct + sum(predicted.data.numpy() == Y.data.numpy())
            total_loss = total_loss + loss
    if (epoch % 50 == 0):
        print('Epoch [%d/%d], Loss: %.4f, Accuracy: %.2f %%'
              % (epoch + 1, num_epochs,
                 total_loss, 100 * correct/total))

# Optional: plotting historical loss from ``all_losses`` during network learning
# Please uncomment me from next line to ``plt.show()`` if you want to plot loss

# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.plot(all_losses)
# plt.show()

"""
Evaluating the Results


"""

train_input = train_data.iloc[:, :input_size]
train_target = train_data.iloc[:, input_size]

inputs = torch.Tensor(train_input.values).float()
targets = torch.Tensor(train_target.values - 1).long()

outputs = net(inputs)
_, predicted = torch.max(outputs, 1)

print('Confusion matrix for training:')
print(plot_confusion(train_input.shape[0], num_classes, predicted.long().data, targets.data))

"""
Step 3: Test the neural network

Pass testing data to the built neural network and get its performance
"""
# get testing data
test_input = test_data.iloc[:, :input_size]
test_target = test_data.iloc[:, input_size]

inputs = torch.Tensor(test_input.values).float()
targets = torch.Tensor(test_target.values - 1).long()

outputs = net(inputs)
_, predicted = torch.max(outputs, 1)

total = predicted.size(0)
correct = predicted.data.numpy() == targets.data.numpy()

print('Testing Accuracy: %.2f %%' % (100 * sum(correct)/total))


"""
Evaluating the Results

To see how well the network performs on different categories, will
create a confusion matrix, indicating for every  (rows)
which class the network (columns).

"""

print('Confusion matrix for testing:')
print(plot_confusion(test_input.shape[0], num_classes, predicted.long().data, targets.data))
