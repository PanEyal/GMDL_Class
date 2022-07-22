# Import Libraries
from sklearn.preprocessing import StandardScaler
import numpy as np
import utils


def PCA(X, components):
    scalar = StandardScaler(with_std=False)
    X_normd = scalar.fit_transform(X)
    correlation = np.cov(X_normd.transpose())
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    p_components = eigenvectors[:, eigenvalues.shape[0] - components:]
    p_components[:, [0, 1]] = p_components[:, [1, 0]]
    return np.matmul(X, p_components).transpose()


def main():
    X, Y = utils.load_MNIST()
    tdr_X = PCA(X, 2)
    utils.scatter_plot(tdr_X, Y, 10)


if __name__ == "__main__":
    main()