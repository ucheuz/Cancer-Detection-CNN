from google.colab import drive
import os

drive.mount('/content/drive')
--------
# edit this path to match where you put your data
path='datasets/histological_data.npz'

full_path=os.path.join('/content/drive/My Drive/', path)

# Load dataset from .npz file
import numpy as np
data = np.load(full_path)
-------
# in case you want to check the data locally
import numpy as np
data = np.load('histological_data.npz')
-------
# Train images and labels
X_train = data['X_train']
y_train = data['y_train'].astype('int')

# Test images and labels
X_test  = data['X_test']
y_test  = data['y_test'].astype('int')

# Print shapes here
print('Training data - images:', X_train.shape)
print('Training data - labels:',y_train.shape)
print('Test data - images:',X_test.shape)
print('Test data - labels:',y_test.shape)
-------
import matplotlib.pyplot as plt

id_images = [4, 5, 6, 7]

plt.figure(figsize=(15, 8))
for i in np.arange(0, 4):
    plt.subplot(1, 4, i+1)
    plt.imshow(X_train[id_images[i], :, :], cmap='gray')
    plt.title('label: ' + str(y_train[id_images[i]]))
-------
import torch

X_train_torch=torch.from_numpy(np.expand_dims(X_train,axis=1)).to(torch.float)
X_test_torch=torch.from_numpy(np.expand_dims(X_test,axis=1)).to(torch.float)

y_train_torch=torch.from_numpy(y_train).to(torch.long)
y_test_torch=torch.from_numpy(y_test).to(torch.long)

print('Training data - newshape:', X_train_torch.shape)
print('Training data - labels:',y_train_torch.shape)
print('Test data - newshape:',X_test_torch.shape)
print('Test data - labels:',y_test_torch.shape)
------
import torch.nn as nn
import torch.nn.functional as F

# will allow network to run on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 4.1.1 create a convolutional layer with kernel size 5 that learns 16 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)

        # maxpool implemented for you
        self.pool = nn.MaxPool2d(2, 2)

        # 4.1.1 create a convolutional layer with kernel size 5 that learns 32 filters
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)

        # 4.1.2 create a linear layer that takes the flattened output from conv2 and learns 100 neuros
        # The dimension of the flattened output is determined by the convolution and pooling operations.
        # Calculation:
        # Input image: 96x96
        # After conv1 (kernel 5, no padding): 96 - 5 + 1 = 92 -> 92x92
        # After pool1: 92 / 2 = 46 -> 46x46
        # After conv2 (kernel 5, no padding): 46 - 5 + 1 = 42 -> 42x42
        # After pool2: 42 / 2 = 21 -> 21x21
        # Flattened size: 32 * 21 * 21
        flattened_size = 32 * 21 * 21
        self.fc1 = nn.Linear(flattened_size, 100) #hint: you need to work out the dimension of the flattened output of conv2 ( i.e. you can print it)

        # 4.1.2 create a linear layer that takes the output from fc2 and learns 2 output neurons
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
       # 4.1.3 implement forward pass
        # Apply the first convolutional layer with ReLU
        x = self.pool(F.relu(self.conv1(x)))

        # Apply the second convolutional layer with ReLU
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output
        x = torch.flatten(x, start_dim=1)

        # Apply the first fully connected layer with ReLU
        x = F.relu(self.fc1(x))

        # Apply the second fully connected layer
        x = self.fc2(x)

        return x



net = Net().to(device)
--------
# loss
loss_fun = nn.CrossEntropyLoss()
loss_fun = loss_fun.to(device)

# optimiser
optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
--------
from sklearn.metrics import accuracy_score

epochs = 500
accuracy=[]
best_val_acc = 0.0

for epoch in range(epochs):

    net.train()
    # send your training data to GPU device
    data = X_train_torch.to(device)
    label = y_train_torch.to(device)

    # 4.3.1: Complete the training loop
    optimizer.zero_grad()
    outputs = net(data)

    err = loss_fun(outputs, label)
    err.backward()

    optimizer.step()

    # validation: evaluate on test set every 10 iterations
    if epoch % 10==0:
        net.eval()

        # 4.3.2: Calculate validation loss and accuracy (use test set)

        data_val = X_test_torch.to(device)
        label_val = y_test_torch.to(device)

        # predict validation outputs
        with torch.no_grad():
            outputs_val = net(data_val)

        # calculate validation error
        val_err = loss_fun(outputs_val, label_val)

        # calculate accuracy
        _, predicted = torch.max(outputs_val, 1)
        acc = accuracy_score(label_val.cpu(), predicted.cpu())

        # Append the validation accuracy to the list
        accuracy.append(acc)

        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}'.format(
            epoch + 1, epochs, err.item(), val_err.item(), accuracy[-1]))

        if acc > best_val_acc:
            best_val_acc = acc
# end of the training loop

# Print the highest validation accuracy that we achieved during training
print('Best Validation Accuracy: {:.4f}'.format(best_val_acc))

# 4.3.3: Plot the validation accuracy over iterations

plt.plot(range(0, epochs, 10), accuracy, label='Validation Accuracy')
plt.title('Validation Accuracy over Iterations')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
