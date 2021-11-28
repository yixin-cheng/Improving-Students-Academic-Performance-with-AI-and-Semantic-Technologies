"""
This file is designed for the LSTM implementation in dropout prediction in ARQ degree.
"""

# import libraries
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from preprocessor import Preprocessor
from feature_selector import Selector


intial_path='data/historicosFinal.csv'
target_path='data/ARQ_clean.csv'
ga_path = 'data/ARQ_final.csv'

"""
perform pre-processing. Uncomment below when you try pre-processing.
"""
# Preprocessor(intial_path,target_path,'ARQ')

"""
perform feature selection. Uncomment below when you try it. Estimated running time: 3hrs+
"""
# Selector(target_path,ga_path)


"""Start Training"""
# store result from each iteration
res=[]

# Device configuration using cuda if exists, cpu otherwise.
if torch.cuda.is_available():
    device = torch.device("cuda:2")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


for i in range (1):
    # Hyper Parameters
    data = pd.read_csv(ga_path, header=None, dtype=float)
    input_size = len(pd.read_csv(ga_path).columns) - 1
    hidden_size = 100
    num_classes = 2
    num_epochs = 20
    num_layers = 2 #number of stacked lstm layers
    batch_size = math.ceil(len(data)*0.8/32) #define the batch by the sequence of each input (32)
    learning_rate = 0.001
    time_step = 1
    data = pd.read_csv(ga_path,header=None,dtype=float)

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
            input = self.data_tensor[index][:-1]
            target = self.data_tensor[index][-1]

            return input, target

        # a function to count samples
        def __len__(self):
            n, _ = self.data_tensor.shape
            return n
    # split data into training set (80%) and testing set (20%)
    train_ind = int(len(data)*0.8)
    train_data=data.iloc[:train_ind]
    test_data=data.iloc[train_ind:]

    # perform dataset loading by adding to Dataloader
    train_data=DataFrameDataset(df=train_data)
    test_data=DataFrameDataset(df=test_data)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=False)

    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             shuffle=False)
    """
    Step 2: Define LSTM

    """
    # Construct LSTM
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(LSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers,batch_first=True,dropout=0.7)  # lstm
            self.fc_1 = nn.Linear(hidden_size, 128)
            self.fc = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()
            self.sigmoid=nn.Sigmoid()


        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            # Propagate input through LSTM
            out, (h_n, h_c) = self.lstm(x, (h0, c0))  # lstm with input, hidden, and internal state
            # print(out.shape)
            # output=self.relu(output)
            out=self.fc_1(out[:,-1,:])
            out = self.relu(out)
            out = self.fc(out)
            return out

    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    print(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # store all losses and accuracy for visualisation
    all_accuracy=[]
    all_losses = []
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            # print('inouts shape: ',inputs.shape)
            # print(features.reshape(features.shape[0],1, input_size).shape)
            inputs = inputs.reshape(-1, time_step, input_size).to(device)
            # print(features.shape)
            labels = labels.long().to(device)

            # forward pass
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            # backward and optimize
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if epoch % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc:{:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),(100 * correct / total)))
                all_losses.append(loss.item())
                all_accuracy.append(100 * correct / total)
    # plot the loss and accuracy as prefer. If not, comment it
    plt.figure()
    plt.plot(all_losses)
    plt.xlabel('epoch*time step')
    plt.ylabel('loss')
    plt.show()
    # plt.savefig('fig/ARQ_loss.jpg')
    plt.figure()
    plt.plot(all_accuracy)
    plt.xlabel('epoch*time step')
    plt.ylabel('accuracy')
    plt.show()
    # plt.savefig('fig/ARQ_acc.jpg')

    """
    Evaluating the Results
    """

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, time_step, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model : {:.4f} %'.format(100 * correct / total))
print(np.average(res))
print(np.max(res))
