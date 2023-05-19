import cv2
import scipy.io
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class BData():
    def __init__(self, skeleton_location, VGG_location):
        self.skeleton = np.array(
            scipy.io.loadmat(skeleton_location)["skeldata"])
        self.VGG = np.array(scipy.io.loadmat(VGG_location)["features"])

    def EDA(self):

        # p = [f"p{i}" for i in range((self.skeleton[:, 0]).shape[0]-3)]

        # index = ["x", "y", "p"]*round((self.skeleton[:, 0].shape[0]-1)/3)
        index = [j+str(i) for j in ["x", "y", "p"]
                 for i in range(round((self.skeleton[:, 0].shape[0]-1)/3))]
        skeleton_df = pd.DataFrame(
            self.skeleton.transpose(), columns=["f", *index])

        ts = skeleton_df.plot.scatter(x='x17', y='y17')
        ts.plot()
        plt.show()
        VGG_df = pd.DataFrame(self.VGG)
        VGG_df.info()

    def SVD(self, r=5):

        U, S, V = np.linalg.svd(self.skeleton, full_matrices=False)
        self.x_aprox = (U[:, :r] * S[:r]) @ Vt[:r, :]


def main():
    data = BData("Data/girosmallveryslow2_openpose_complete.mat",
                 "Data/girosmallveryslow2_vggfeatures.mat")
    data.EDA()
    data.SVD()


if __name__ == '__main__':
    main()
