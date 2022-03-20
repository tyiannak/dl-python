import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import f1_score


def evaluate_model(n_model, test_data, test_labels):
    X_test_var = Variable(torch.FloatTensor(test_data), requires_grad=False) 
    with torch.no_grad():
        test_result = n_model(X_test_var)
    values, labels = torch.max(test_result, 1)
    y_pred = labels.data.numpy()
    return f1_score(y_pred, test_labels)


n_samples_per_class = 200
indepedence_rate = 0.5
dimensions = 20

n_samples = n_samples_per_class
D = dimensions # dimensions
mean_1 = np.random.rand(D)
mean_2 = mean_1 + 1 * np.random.random(D)
# create the covarience matrix (that )
cov = np.eye(D) + np.random.random((D, D)) / indepedence_rate
cov[cov>1] = 1
cov[cov<0] = 0
cov = (cov + cov.T) / 2
X = np.concatenate([np.random.multivariate_normal(mean_1, cov, n_samples),
                    np.random.multivariate_normal(mean_2, cov, n_samples)])    
y = np.concatenate([np.zeros((n_samples,)), np.ones((n_samples, ))])

# Split to train/test
X_train = X[::2, :]
y_train = y[::2]
X_test = X[1::2, :]
y_test = y[1::2]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dimensions, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
model = Net()
print(model)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

batch_size = 32
n_epochs = 500
batch_no = len(X_train) // batch_size

train_loss_min = np.Inf
for epoch in range(n_epochs):
    train_loss = 0  # should be this initalized here?
    for i in range(batch_no):
        start = i * batch_size
        end   = start + batch_size
        x_var = Variable(torch.FloatTensor(X_train[start:end]))
        y_var = Variable(torch.LongTensor(y_train[start:end])) 
        
        optimizer.zero_grad()
        output = model(x_var)
        loss   = criterion(output, y_var)
        loss.backward()
        optimizer.step()
        
        values, labels = torch.max(output, 1)
        f1_train = f1_score(labels.data.numpy(), y_train[start:end])
        train_loss += loss.item()*batch_size
    
    train_loss = train_loss / len(X_train)
    if train_loss <= train_loss_min:
        print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(train_loss_min,train_loss))
        torch.save(model.state_dict(), "model.pt")
        train_loss_min = train_loss
    
    if epoch % 20 == 0:
        print('')
        f1_test = evaluate_model(model, X_test, y_test)
        print("Epoch: {} \tTrain Loss: {} \tTrain Accuracy: {} \tTest Accuracy: {}".format(epoch+1, train_loss, f1_train, f1_test))
print('Training Ended! ')

