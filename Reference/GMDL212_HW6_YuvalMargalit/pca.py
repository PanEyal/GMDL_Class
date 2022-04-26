# Import Libraries
import sklearn
import scipy
import utils
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def myPCA(X, components):
    scalar = StandardScaler(with_std=False)
    X_normd = scalar.fit_transform(X)
    correlation = np.cov(np.transpose(X_normd))
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    p_components = eigenvectors[:, eigenvalues.shape[0] - components:]
    p_components[:,[0,1]] = p_components[:,[1,0]]
    return np.matmul(X, p_components)


def main():
    X, Y = utils.load_data("train.csv")
    twod_X = myPCA(X.to_numpy(), 2)
    utils.scatter_plot(twod_X, Y, 10)


if __name__ == "__main__":
    main()
