# Import Libraries
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import utils


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            nn.ReLU(),
            torch.nn.Linear(128, 64),
            nn.ReLU(),
            torch.nn.Linear(64, 8),
            nn.ReLU(),
            torch.nn.Linear(8, 2),
            nn.ReLU())
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            nn.ReLU(),
            torch.nn.Linear(8, 64),
            nn.ReLU(),
            torch.nn.Linear(64, 128),
            nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            nn.ReLU())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)


class linear_autoencoder(nn.Module):
    def __init__(self):
        super(linear_autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, 8),
            torch.nn.Linear(8, 2))
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.Linear(8, 64),
            torch.nn.Linear(64, 128),
            torch.nn.Linear(128, 28 * 28))


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)


def main():
    X, Y = utils.load_data("train.csv")
    # X, Y = utils.load_data("..\\train.csv")
    standardized_scalar = StandardScaler(with_std=False)
    X = standardized_scalar.fit_transform(X)
    X = np.array(X)
    Y = np.array(Y).reshape((-1, 1))
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    model = autoencoder()
    criterion = nn.MSELoss()
    optimaizer = torch.optim.Adam(model.parameters(), lr=0.001)
    utils.train(30, train_loader, model, criterion, optimaizer)
    embeddings, labels = utils.get_embedding(model, train_loader)
    utils.scatter_plot(embeddings, labels, 10)
    model=linear_autoencoder()
    criterion = nn.MSELoss()
    optimaizer = torch.optim.Adam(model.parameters(), lr=0.001)
    utils.train(30, train_loader, model, criterion, optimaizer)
    embeddings, labels = utils.get_embedding(model, train_loader)
    utils.scatter_plot(embeddings, labels, 10)

if __name__ == "__main__":
    main()
