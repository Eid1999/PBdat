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
        p = [f"p{i}" for i in range((self.skeleton[:, 0]).shape[0]-3)]
        index = ["f", "x", "y", *p]
        skeleton_df = pd.DataFrame(self.skeleton.transpose(), columns=index)

        ts = skeleton_df.plot.scatter(x='x', y='y')
        ts.plot()
        plt.show()

    def PCA(self):
        pass


def main():
    data = BData("Data/girosmallveryslow2_openpose_complete.mat",
                 "Data/girosmallveryslow2_vggfeatures.mat")
    data.EDA()
    data.PCA()


if __name__ == '__main__':
    main()
