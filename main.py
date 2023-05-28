# %%
import scipy.io
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# %%


class BData():
    def __init__(self, skeleton_location, VGG_location):
        print("Loading data")
        self.skelly = []
        self.skeleton = np.array(
            scipy.io.loadmat(skeleton_location)["skeldata"])[:, 1:].transpose()
        self.VGG = np.array(scipy.io.loadmat(
            VGG_location)["features"]).transpose()

    def center_data(self, df):
        print()
        print("Centering data")
        # center the data
        df = df - df.mean()
        return df

    def EDA_VGG(self):
        print()
        print("VGG EDA")
        self.VGG_df = pd.DataFrame(self.VGG)
        print(self.VGG_df.describe())

    def EDA_skelly(self):
        print()
        print("Skelly EDA")
        # index obtains the indexes of the columns by concatenating the strings "x", "y" and "p" with the numbers 1 to 18
        # and then concatenating the strings with the numbers 1 to 6 and putting them in a list
        index = [j+str(i+1) for j in ["x", "y", "p"]
                 for i in range(int((self.skeleton[0, :].shape[0]-1)/3))]
        # skeleton_df is a dataframe with the skeleton data and the indexes are the columns
        self.skeleton_df = pd.DataFrame(
            self.skeleton, columns=["f", *index])
        # describe the data
        # plot the data
        max_frame = max([
            self.skeleton_df[self.skeleton_df.f == i].shape[0] for i in range(int(self.skeleton_df["f"].max()))])
        # ts = self.skeleton_df.plot.scatter(x='x1', y='y1')
        # ts.plot()
        # plt.show()
        skelly_aux = []
        for i in range(18):
            # drop the p of the joints
            self.skeleton_df = self.skeleton_df.drop(
                columns=["p"+str(i+1)])
        skelly_aux = []
        for i in range(int(self.skeleton_df["f"].max())):
            # skelly is a dataframe with the skeleton data of the frame i
            skelly = self.skeleton_df[self.skeleton_df.f == i].drop(
                columns="f")
            middle_point = np.array([skelly["x1"],
                                     skelly["y1"]])
            if middle_point.shape[1] == 0:
                n = 0
                mean = [0]*34
                autocorrolation = [0]*34

            else:
                n = len(middle_point)
                skelly = skelly.drop(
                    columns=["x1", "y1"]).transpose().values.reshape(
                    -1, middle_point.shape[1]*2)
                skelly = ((skelly-middle_point.reshape(1, -1))
                          ).transpose().reshape(-1, 34)
                mean = skelly.mean(axis=0)
                # autocorrolation = skelly.apply(
                #    lambda col: col.autocorr(1), axis=0){i}")
            #     mean_string.append(f"mean_Y{i}")
            #     autocorrolation_string.append(f"autocorr_X{i}")
            #     autocorrolation_string.append(f"autocorr_Y{i}")

            # # pad the data with 0s to have the same number of frames
            # if skelly.shape[0] < max_frame:
            #     skelly = np.pad(
            #         skelly, ((0, max_frame - skelly.shape[0]), (0, 0)), mode='constant')

                # mean_string = []
            autocorrolation_string = []

            # for i in range(1, 19):
            #     mean.append(skelly[f"x{i}"].mean())
            #     mean.append(skelly[f"y{i}"].mean())
            #     autocorrolation.append(skelly[f"x{i}"].autocorr())
            #     autocorrolation.append(skelly[f"y{i}"].autocorr())
            #     mean_string.append(f"mean_X{i}")
            #     mean_string.append(f"mean_Y{i}")
            #     autocorrolation_string.append(f"autocorr_X{i}")
            #     autocorrolation_string.append(f"autocorr_Y{i}")

            # # pad the data with 0s to have the same number of frames
            # if skelly.shape[0] < max_frame:
            #     skelly = np.pad(
            #         skelly, ((0, max_frame - skelly.shape[0]), (0, 0)), mode='constant')

            # Flatten the matrices into vectors
            # vector = np.reshape(skelly, -1)
            skelly_aux.append([n, *mean])

        print(self.skeleton_df.describe())

        # skeleton_df is a dataframe with the skeleton data of all frames
        self.skeleton_df = pd.DataFrame(
            skelly_aux)

        # correlation = self.skeleton_df.corr()
        # sns.heatmap(correlation, xticklabels=correlation.columns,
        # yticklabels=correlation.index, annot=True)
        # plt.show()

    def PCA_VGG(self):
        print()
        print("PCA VGG")
        print(np.linalg.matrix_rank(self.VGG_df))

        steps = [('scaling', StandardScaler()),
                 ('pca', PCA(n_components=0.9))]
        # create the pipeline
        pipeline = Pipeline(steps)
        # fit the pipeline
        pipeline.fit(self.VGG_df)
        # transform the data
        reduced = pipeline.inverse_transform(
            pipeline.transform(self.VGG_df))
        # create a dataframe with the reduced data
        self.VGG_df = pd.DataFrame(reduced)
        # rank of vgg_df
        print(np.linalg.matrix_rank(self.VGG_df))

    def PCA_skelly(self):
        print()
        print("PCA Skelly")
        print(np.linalg.matrix_rank(self.skeleton_df))
        steps = [('scaling', StandardScaler()),
                 ('pca', PCA(n_components=0.90))]
        pipeline = Pipeline(steps)
        pipeline.fit(self.skeleton_df)
        reduced = pipeline.inverse_transform(
            pipeline.transform(self.skeleton_df))
        self.skeleton_df = pd.DataFrame(reduced)
        print(np.linalg.matrix_rank(self.skeleton_df))

    def kmeans(self, df):
        print()
        print("Kmeans")
        # apply kmeans with 6 clusters
        km = KMeans(n_clusters=6, n_init="auto").fit(df)
        # get the centroids of each cluster
        clusters = km.cluster_centers_
        # plot the data
        plt.figure(num=None, figsize=(10, 10), dpi=100,
                   facecolor='w', edgecolor='k')
        # plt.scatter(df[:][0], df[:][1], alpha=0.25, s=100, cmap='rainbow')
        # plt.scatter(clusters[:, 0], clusters[:, 1], s=100000, alpha=0.30)
        # plot each cluster with a different color
        # 3d plot
        ax = plt.subplot(111, projection='3d')
        # another plot but 2d
        for i in range(6):
            ax.scatter(df[km.labels_ == i][0], df[km.labels_ == i][1],
                       df[km.labels_ == i][2], alpha=0.25, s=100, cmap='rainbow')
            # plot the centroid of each cluster
            ax.scatter(clusters[i, 0], clusters[i, 1],
                       clusters[i, 2], s=100, alpha=0.30, color="black")
            print("The centroid of the cluster number " + str(i) +
                  " is " + str(clusters[i, 0]) + " " + str(clusters[i, 1]))
            # find the frame with the minimum distance to the centroid
            min_dista = 1000000
            min_frame = 0
            for j in range(df[km.labels_ == i][0].shape[0]):
                dista = np.sqrt((df[km.labels_ == i][0].iloc[j] - clusters[i, 0])**2 + (df[km.labels_ == i]
                                [1].iloc[j] - clusters[i, 1])**2 + (df[km.labels_ == i][2].iloc[j] - clusters[i, 2])**2)
                if dista < min_dista:
                    min_dista = dista
                    min_frame = j
            print("The frame with the minimum distance to the centroid is " + str(min_frame) + " so the frame is in the minute " +
                  str(round(min_frame/60)) + " and second " + str(min_frame % 60) + " of the video")
        plt.show()

    def t_SNE(self, df):
        print()
        print("t-SNE")
        # calculate the number of components of the t-SNE algorithm
        tsne = TSNE(n_components=2, perplexity=50, verbose=2).fit_transform(df)
        # plot the data
        plt.figure(num=None, figsize=(10, 10), dpi=100,
                   facecolor='w', edgecolor='k')
        # ax = plt.subplot(111, projection='3d')
        plt.scatter(tsne[:, 0], tsne[:, 1], alpha=0.25, s=100, cmap='rainbow')
        plt.show()


# %%
def main():
    data = BData("Data/girosmallveryslow2_openpose_complete.mat",
                 "Data/girosmallveryslow2_vggfeatures.mat")
    data.EDA_VGG()
    data.PCA_VGG()
    data.VGG_df = data.center_data(data.VGG_df)
    data.kmeans(data.VGG_df)
    data.EDA_skelly()
    data.PCA_skelly()
    data.skeleton_df = data.center_data(data.skeleton_df)
    data.kmeans(data.skeleton_df)
    data.t_SNE(data.VGG_df)
    data.t_SNE(data.skeleton_df)


if __name__ == '__main__':
    main()
