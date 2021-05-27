__author__ = 'SherlockLiao'

import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image


from pyAudioAnalysis import MidTermFeatures as mtf
import numpy as np
import pickle
import os.path

if os.path.isfile('data.pkl'):
    with open('data.pkl','rb') as f:
        mid_term_features_2 = pickle.load(f)
        wav_file_list2 = pickle.load(f)
else:
    with open('data.pkl','wb') as f:
        mid_term_features, wav_file_list2, mid_feature_names = mtf.directory_feature_extraction('audio', 1, 1, 0.2, 0.2)
        mid_term_features = mid_term_features[:, 0:128]
        m = mid_term_features.mean(axis=0)
        s = np.std(mid_term_features, axis = 0)
        mid_term_features_2 = (mid_term_features - m) / s
        pickle.dump(mid_term_features_2, f)
        pickle.dump(wav_file_list2, f)
x = torch.tensor(mid_term_features_2, dtype=torch.float32)


num_epochs = 200
batch_size = 128
learning_rate = 1e-3

dataset = TensorDataset(x)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5 )

for epoch in range(num_epochs):
    for data in dataloader:
        img = data
        img = img[0]
#        print(img)
#        img = img.view(img.size(0), -1)
#        print(img.shape)
        img = Variable(img).to(device)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    if epoch % 10 == 0:
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss))

torch.save(model.state_dict(), './sim_autoencoder.pth')

from torch.autograd import Variable as V
x_=[]
y_=[]
import pandas
data_to_plot = []

# produce the code:
X = []
for i, d in enumerate(dataset):
    pred = model.encoder(d[0]).data.cpu().numpy()
    X.append(pred)
X = np.array(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = pca.fit_transform(X)
weights = np.random.rand(16, 2)
#weights = np.random.rand(128, 2)
for i, d in enumerate(dataset):
    if i % 20 == 0:
#        pred = model.encoder(d[0]).data.cpu().numpy()
#        pred = np.dot(pred, weights)
#        pred = np.dot(mid_term_features_2[i, :], weights)
        pred = X[i, :]
        print(pred.shape)
        print(wav_file_list2[i][5:15])
        print(pred)
        data_to_plot.append({'x': pred[0], 'y': pred[1], 
            'name': wav_file_list2[i][6:11]})

print(data_to_plot)
df = pandas.DataFrame(data_to_plot)
import plotly.express as px
fig = px.scatter(df, x="x", y="y", text="name", size_max=60)
fig.update_traces(textposition='top center')

fig.update_layout(
    height=800,
    title_text='GDP and Life Expectancy (Americas, 2007)'
)

fig.show()
#    #loss = criterion(pred, x)
#    dimension=_.data.numpy()
#    x_.append(dimension[0])
#    y_.append(dimension[1])
