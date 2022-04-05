import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# loading dataset
train = pd.read_csv('data/apparel/train_LbELtWX/train.csv')
test = pd.read_csv('data/apparel/test_ScVgIM0/test.csv')
train = train[::5]
print(train)

# loading training images
train_img = []
for img_name in tqdm(train['id']):
    image_path = 'data/apparel/train_LbELtWX/train/' + str(img_name) + '.png'
    img = imread(image_path, as_gray=True)      # read the image
    img /= 255.0    # normalize pixels
    img = img.astype('float32')    # convert int to float
    train_img.append(img)

train_x = np.array(train_img)
train_y = train['label'].values

# split data:
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1)
print((train_x.shape, train_y.shape), (val_x.shape, val_y.shape))
train_x = torch.from_numpy(train_x.reshape(train_x.shape[0], 1, 28, 28))
train_y = torch.from_numpy(train_y.astype(int))
val_x = torch.from_numpy(val_x.reshape(val_x.shape[0], 1, 28, 28))
val_y = torch.from_numpy(val_y.astype(int))
print(val_x.shape, val_y.shape)

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

model = Net()
optimizer = Adam(model.parameters(), lr=0.01)
criterion = CrossEntropyLoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()    
print(model)

def train(epoch):
    model.train()
    tr_loss = 0
    x_train, y_train = Variable(train_x), Variable(train_y)
    x_val, y_val = Variable(val_x), Variable(val_y)

    if torch.cuda.is_available():      # convert data to GPU format
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    optimizer.zero_grad()      # clear the Gradients
    output_train = model(x_train)   # prediction for training set
    output_val = model(x_val)   # prediction for validation set

    # compute losses
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # backprop and update weights:
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)

# do training:
n_epochs = 50
train_losses = []
val_losses = []
for epoch in range(n_epochs):
    train(epoch)

# prediction for training set
with torch.no_grad():
    output = model(train_x)

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on training set
print(accuracy_score(train_y, predictions))