import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets

### load the MNIST data set and corresponding labels
def load_MNIST(batch_size_train=10000, random_seed=False):
    train_set = datasets.MNIST('./data', train=True, download=True)
    X = train_set.data.numpy()
    Y = train_set.targets.numpy()
    choice = np.random.choice(range(X.shape[0]), size=batch_size_train, replace=False)
    return X[choice], Y[choice]

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
    

### FOR THE AUTOENCODER PART
def train(num_epochs,dataloader,model,criterion,optimizer):
    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            # ===================forward=====================
            output,_ = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))

def get_embedding(model,dataloader):
    model.eval()
    labels = np.zeros((0,))
    embeddings = np.zeros((2,0))
    for data in dataloader:
        X,Y = data
        with torch.no_grad():
            _,code = model(X)
        embeddings = np.hstack([embeddings,code.numpy().T])
        labels = np.hstack([labels,np.squeeze(Y.numpy())])
    return embeddings,labels