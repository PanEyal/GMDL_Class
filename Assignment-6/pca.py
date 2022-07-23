# Import Libraries
from sklearn.preprocessing import StandardScaler
import numpy as np
import utils


def PCA(X, components):
    scalar = StandardScaler(with_std=False)
    X_normd = scalar.fit_transform(X)
    cov_mat = np.cov(X_normd.transpose())
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
    reducing_size = eigenvalues.shape[0] - components
    pc = eigenvectors[:, reducing_size:]
    pc[:, [0, 1]] = pc[:, [1, 0]]
    return np.matmul(X, pc).transpose()


def main():
    X, Y = utils.load_MNIST(random_seed=1)
    tdr_X = PCA(X, 2)
    utils.scatter_plot(tdr_X, Y, 10)


if __name__ == "__main__":
    main()