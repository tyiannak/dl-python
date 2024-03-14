from faulthandler import is_enabled
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import f1_score
from pyAudioAnalysis import MidTermFeatures as mt
from pyAudioAnalysis import audioTrainTest as aT
import numpy as np
import os


def evaluate_model(n_model, test_data, test_labels):
    X_test_var = torch.tensor(test_data, device=my_device, dtype=torch.float32)
    with torch.no_grad():
        test_result = n_model(X_test_var)
    values, labels = torch.max(test_result, 1)
    y_pred = labels.data.cpu().numpy()
    return f1_score(y_pred, test_labels)

# get audio features for the two audio classes using pyAudioAnalysis:
if os.path.isfile("features.npy"):
    with open('features.npy', 'rb') as f:
        X = np.load(f)
        y = np.load(f)
else:
    features, class_names, file_names = mt.multiple_directory_feature_extraction(["audio/speech", "audio/noise"], 1, 1, 0.1, 0.1, False)
    X, y = aT.features_to_matrix(features)
    with open('features.npy', 'wb') as f:
        np.save(f, np.array(X))
        np.save(f, np.array(y))

dimensions = X.shape[1]

# Split to train/test (naive)
X_train = X[::2, :]
y_train = y[::2]
X_test = X[1::2, :]
y_test = y[1::2]

# define the network architecture:
n_nodes = 256

my_device = torch.device("cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dimensions, n_nodes)
        self.fc1_bn = nn.BatchNorm1d(n_nodes)

        self.fc2 = nn.Linear(n_nodes, n_nodes)
        self.fc2_bn = nn.BatchNorm1d(n_nodes)

        self.fc3 = nn.Linear(n_nodes, n_nodes)
        self.fc3_bn = nn.BatchNorm1d(n_nodes)

        self.fc4 = nn.Linear(n_nodes, 2)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return x


model = Net().to(my_device)
print(next(model.parameters()).is_mps)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

batch_size = 16
n_epochs = 500
batch_no = len(X_train) // batch_size

f1_test_max = -np.Inf
f1s = []
patience = 50
since_last_best = 0
for epoch in range(n_epochs):
    train_loss = 0
    for i in range(batch_no):  # for each batch (TODO Use dataloaders in another example!)
        # get batch data:
        start = i * batch_size
        end = start + batch_size
        x_var = Variable(torch.FloatTensor(X_train[start:end])).to(my_device)
        y_var = Variable(torch.LongTensor(y_train[start:end])).to(my_device)

        # clear the gradients:
        optimizer.zero_grad()

        # forward pass:
        output = model(x_var).to(my_device)

        # calculate loss on training data
        loss = criterion(output, y_var)

        # calculate gradients
        loss.backward()

        # update weights (gradient descent):
        optimizer.step()

        # update loss:
        train_loss += loss.item() * batch_size

        # get training f1: 
        values, labels = torch.max(output, 1)
#        f1_train = f1_score(labels.data.numpy(), y_train[start:end])

    train_loss = train_loss / len(X_train)
    f1_test = evaluate_model(model, X_test, y_test)
    f1s.append(f1_test)
    since_last_best += 1
    if f1_test >= f1_test_max:
        is_best = " (best) "
        torch.save(model.state_dict(), "model.pt")
        f1_test_max = f1_test
        since_last_best = 0
    else:
        is_best = ""
    if since_last_best > patience:
        break

#    print(f'Epoch {epoch} - Training loss {train_loss:.5f} - Training F1 {100*f1_train:.2f} - {is_best}')
    if epoch % 1 == 0 or len(is_best) > 0:
        print(since_last_best)
        print(f'Epoch {epoch} - Training loss {train_loss:.5f} - Eval F1 {100*f1_test:.2f} {is_best}')

print('Training Ended! ')

import matplotlib.pyplot as plt
plt.plot(f1s)
plt.show()
