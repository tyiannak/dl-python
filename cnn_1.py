import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def evaluate_model(n_model, data, labels):
    # prediction for training set
    device = next(model.parameters()).device
    with torch.no_grad():
        output = n_model(data.to(device))
    softmax = torch.exp(output)
    #prob = list(softmax.numpy())
    predictions = torch.argmax(softmax, -1)
    return f1_score(labels.cpu().data.numpy(), predictions.cpu().data.numpy(), average='macro')


def read_and_preprocess_data(csv_file, data_path):
    data = pd.read_csv(csv_file)
    data = data[::5]
    data_img = []
    for img_name in tqdm(data['id']):
        image_path = data_path + str(img_name) + '.png'
        img = imread(image_path, as_gray=True) / 255.0  # read image & normalize
        img = img.astype('float32')    # convert int to float
        data_img.append(img)
    data_x = np.array(data_img)
    data_y = data['label'].values
    return data_x, data_y

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(7 * 7 * 32, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def train(model, X_train, y_train, X_val, y_val, n_epochs = 20):

    train_losses, val_losses, train_f1, val_f1 = [], [], [], []

    for e in range(n_epochs):
        model.train()
        tr_loss = 0
        x_train, y_train = Variable(X_train), Variable(y_train)
        x_val, y_val = Variable(X_val), Variable(y_val)

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
        train_losses.append(loss_train.cpu().detach().numpy())
        val_losses.append(loss_val.cpu().detach().numpy())

        # backprop and update weights:
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()
        print('Epoch : ', e+1, '\t', 'loss :', loss_val)
        train_f1.append(evaluate_model(model, X_train, y_train))
        val_f1.append(evaluate_model(model, X_val, y_val))
    import matplotlib.pyplot as plt
    plt.subplot(2,1,1)
    plt.plot(range(n_epochs), train_losses, '--')
    plt.plot(range(n_epochs), val_losses)
    plt.subplot(2,1,2)
    plt.plot(range(n_epochs), train_f1, '--')
    plt.plot(range(n_epochs), val_f1)
    plt.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# read data:
train_x, train_y = read_and_preprocess_data('train_LbELtWX/train.csv', 
                                            'train_LbELtWX/train/')

# split data:
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1)
train_x = torch.from_numpy(train_x.reshape(train_x.shape[0], 1, 28, 28)).to(device)
train_y = torch.from_numpy(train_y.astype(int)).to(device)
val_x = torch.from_numpy(val_x.reshape(val_x.shape[0], 1, 28, 28)).to(device)
val_y = torch.from_numpy(val_y.astype(int)).to(device)

model = Net()
optimizer = Adam(model.parameters(), lr=0.01)
criterion = CrossEntropyLoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
train(model, train_x, train_y, val_x, val_y, n_epochs=200)

