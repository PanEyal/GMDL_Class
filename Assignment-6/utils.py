import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets
from tqdm import tqdm

### load the MNIST data set and corresponding labels
def load_MNIST(batch_size_train=10000, random_seed=False):
    train_set = datasets.MNIST('./data', train=True, download=True)
    X = train_set.data.numpy()
    Y = train_set.targets.numpy()
    choice = np.random.choice(range(X.shape[0]), size=batch_size_train, replace=False)
    X = X[choice]
    Y = Y[choice]
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2] ), Y

###
# plot a scatter plot of coordinates with labels labels
# the data contain k classes
###
def scatter_plot(coordinates,labels,k):
    fig, ax = plt.subplots()
    for i in range(k):
        idx = labels == i
        data = coordinates[:,idx]
        ax.scatter(data[0],data[1],label=str(i),alpha=0.3,s=10)
    ax.legend(markerscale=2)
    plt.show()

###
# plot the losses graph
###
def losses_plot(autoencoder_losses, linear_autoencoder_losses):
    plt.title('Loss on models')
    plt.plot(autoencoder_losses, label='autoencoder_loss')
    plt.plot(linear_autoencoder_losses, label='linear_autoencoder_loss')
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

### FOR THE AUTOENCODER PART
def train(num_epochs,dataloader,model,criterion,optimizer):
    loss_array = []
    for epoch in tqdm(range(num_epochs)):
        for data in dataloader:
            img, _ = data
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        loss_array.append(loss.item())
    return loss_array

def get_embedding(model,dataloader):
    model.eval()
    labels = np.zeros((0,))
    embeddings = np.zeros((2,0))
    for data in dataloader:
        X,Y = data
        with torch.no_grad():
            code = model.encode(X)
        embeddings = np.hstack([embeddings,code.numpy().T])
        labels = np.hstack([labels,np.squeeze(Y.numpy())])
    return embeddings,labels