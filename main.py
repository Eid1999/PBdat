import cv2
import scipy.io
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans


class BData():
    def __init__(self, skeleton_location, VGG_location):
        self.skelly = []
        self.skeleton = np.array(
            scipy.io.loadmat(skeleton_location)["skeldata"])[:, 1:].transpose()
        self.VGG = np.array(scipy.io.loadmat(
            VGG_location)["features"]).transpose()

    def EDA_VGG(self):
        self.VGG_df = pd.DataFrame(self.VGG)

        # print(VGG_df.describe())

    def EDA_skelly(self):
        index = [j+str(i+1) for j in ["x", "y", "p"]
                 for i in range(int((self.skeleton[0, :].shape[0]-1)/3))]
        self.skeleton_df = pd.DataFrame(
            self.skeleton, columns=["f", *index])
        max_frame = max([
            self.skeleton_df[self.skeleton_df.f == i].shape[0] for i in range(int(self.skeleton_df["f"].max()))])
        # ts = self.skeleton_df.plot.scatter(x='x1', y='y1')
        # ts.plot()
        # plt.show()
        skelly_aux = []
        for i in range(int(self.skeleton_df["f"].max())):
            skelly = self.skeleton_df[self.skeleton_df.f == i].drop(
                columns="f")

            if skelly.shape[0] < max_frame:
                skelly = np.pad(
                    skelly, ((0, max_frame - skelly.shape[0]), (0, 0)), mode='constant')

            # Flatten the matrices into vectors
            vector = np.reshape(skelly, -1)
            skelly_aux.append(vector)

        self.skeleton_df = pd.DataFrame(skelly_aux).squeeze()

        # correlation = self.skeleton_df.corr()
        # sns.heatmap(correlation, xticklabels=correlation.columns,
        # yticklabels=correlation.index, annot=True)
        # plt.show()

    def PCA_VGG(self):
        steps = [('scaling', StandardScaler()),
                 ('pca', PCA(n_components=0.85))]

        pipeline = Pipeline(steps)
        pipeline.fit(self.VGG_df)
        reduced = pipeline.inverse_transform(
            pipeline.transform(self.VGG_df))

        self.VGG_df = pd.DataFrame(reduced[:, 0:2])

    def PCA_skelly(self):

        skelly = 0
        steps = [('scaling', StandardScaler()),
                 ('pca', PCA(n_components=0.85))]
        pipeline = Pipeline(steps)
        pipeline.fit(self.skeleton_df)

        self.skeleton_df = pipeline.inverse_transform(
            pipeline.transform(self.skeleton_df))

    def kmeans(self, df):
        km = KMeans(n_clusters=2, n_init="auto").fit(df)
        clusters = km.cluster_centers_
        plt.figure(num=None, figsize=(10, 10), dpi=100,
                   facecolor='w', edgecolor='k')
        plt.scatter(df[:][0], df[:][1], alpha=0.25, s=100, color='red')
        plt.scatter(clusters[:, 0], clusters[:, 1], s=100000, alpha=0.30)

        # print(y)
        # plt.scatter(reduced[0, :], reduced[1, :])
        plt.show()


def main():
    data = BData("Data/girosmallveryslow2_openpose.mat",
                 "Data/girosmallveryslow2_vggfeatures.mat")
    data.EDA_VGG()
    data.PCA_VGG()
    data.kmeans(data.VGG_df)
    data.EDA_skelly()
    data.PCA_skelly()
    data.kmeans(data.skeleton_df)


if __name__ == '__main__':
    main()