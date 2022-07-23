# Import Libraries
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import utils


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.name = "autoencoder"

        # Encoder: affine function
        self.encode = torch.nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU()
        )
        # Decoder: affine function
        self.decode = torch.nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.ReLU()
        )

    def forward(self, X):
        out = self.encode(X)
        out = self.decode(out)

        return out


class linear_autoencoder(nn.Module):
    def __init__(self):
        super(linear_autoencoder, self).__init__()
        self.name = "linear_autoencoder"

        # Encoder: affine function
        self.encode = torch.nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 8),
            nn.Linear(8, 4),
            nn.Linear(4, 2)
        )
        # Decoder: affine function
        self.decode = torch.nn.Sequential(
            nn.Linear(2, 4),
            nn.Linear(4, 8),
            nn.Linear(8, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 28 * 28)
        )

    def forward(self, X):
        out = self.encode(X)
        out = self.decode(out)

        return out


def main():
    X, Y = utils.load_MNIST()
    scalar = StandardScaler(with_std=False)
    X = scalar.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

    # Computer Exercise 4
    model = autoencoder()
    criterion = nn.MSELoss()
    optimaizer = torch.optim.Adam(model.parameters(), lr=0.001)
    autoencoder_losses = utils.train(30, train_loader, model, criterion, optimaizer)
    embeddings, labels = utils.get_embedding(model, train_loader)
    utils.scatter_plot(embeddings, labels, 10)

    # Computer Exercise 5
    model = linear_autoencoder()
    criterion = nn.MSELoss()
    optimaizer = torch.optim.Adam(model.parameters(), lr=0.001)
    linear_autoencoder_losses = utils.train(30, train_loader, model, criterion, optimaizer)
    embeddings, labels = utils.get_embedding(model, train_loader)
    utils.scatter_plot(embeddings, labels, 10)
    utils.losses_plot(autoencoder_losses, linear_autoencoder_losses)

if __name__ == "__main__":
    main()
