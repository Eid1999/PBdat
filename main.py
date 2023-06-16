

import os
import cv2
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
#from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
import argparse
from yellowbrick.cluster import KElbowVisualizer





class BData():
    def __init__(self, skeleton_location, VGG_location, video_location):
        print("Loading data")
        

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
        index = [j+str(i+1) for i in range(int((self.skeleton[0, :].shape[0]-1)/3))
                 for j in ["x", "y", "p"]]
        # skeleton_df is a dataframe with the skeleton data and the indexes are the columns
        self.skeleton_df = pd.DataFrame(
            self.skeleton, columns=["f", *index])
        # describe the data
        
        self.Outliers()    
        print(self.skeleton_df.describe())

    def Outliers(self):
        j=0
        os.makedirs("Outliers", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, layers = self.video[1].shape
        # Codec (e.g., 'XVID', 'MJPG', 'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 2
        video = cv2.VideoWriter(f"Outliers/video.mp4", fourcc, fps, (width, height))
        for i in range(1,int(max(self.skeleton_df["f"]))):
            if self.skeleton_df[self.skeleton_df.f==i].empty:
                self.skeleton_df=self.skeleton_df.loc[self.skeleton_df['f'] != i]
                video.write(self.video[i-j])
                self.video.pop(i-j)
                j+=1
        video.release()
    def missing_data(self):
        print()
        print("Skelly Missing Data")
        index1 = [j+str(i+1) for i in range(int((self.skeleton[0, :].shape[0]-1)/3))
                  for j in ["x", "y"]]
        self.skeleton_df[self.skeleton_df.drop(
            columns="f") == 0] = np.nan
        imputerX = KNNImputer(n_neighbors=2)
        imputerY = KNNImputer(n_neighbors=2)
        if self.skeleton_df.isnull().values.any():

            X_values=self.skeleton_df.loc[:,index1[::2]]
            Y_values=self.skeleton_df.loc[:,index1[1::2]]
            self.skeleton_df.loc[:,index1[::2]] = imputerX.fit_transform(X_values)
            self.skeleton_df.loc[:,index1[1::2]]= imputerY.fit_transform(Y_values)
            # self.skeleton_df.loc[:,index1[::1]] = imputerX.fit_transform(self.skeleton_df.loc[:,index1[::1]])
    def data_manipulation(self):
        print("\nSkelly Data Manpulation")
        skelly_aux = []
        for i in range(int((self.skeleton[0, :].shape[0]-1)/3)):
            # drop the p of the joints
            self.skeleton_df = self.skeleton_df.drop(
                columns=["p"+str(i+1)])

        
        for i in range(int(self.skeleton_df["f"].max())):
            # skelly is a dataframe with the skeleton data of the frame i
            skelly = self.skeleton_df[self.skeleton_df.f == i].drop(
                columns="f")
                
            middle_point = np.array([skelly["x2"],
                                     skelly["y2"]])


            if middle_point.shape[1] == 0:
                n = 0
                mean = [0]*34
                variance = [0]*34

            else:
                n = len(middle_point)
                skelly = skelly.drop(
                    columns=["x2", "y2"]).transpose().values.reshape(
                    -1, middle_point.shape[1]*2)
                skelly = ((skelly-middle_point.reshape(1, -1))
                          ).transpose().reshape(-1, 34)
                mean = skelly.mean(axis=0)


                variance = skelly.var(axis=0)

            # autocorrolation_string = []

            skelly_aux.append([n, *mean, *variance])

        self.skeleton_df = pd.DataFrame(
            skelly_aux)
            

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
        self.VGG_pca = pd.DataFrame(reduced)
        # rank of vgg_df
        print(np.linalg.matrix_rank(self.VGG_pca))
        self.kmeans(self.VGG_pca, plot_type="3d",n_clusters=6)
        #self.n_clustering_number(self.VGG_pca)

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
        self.skeleton_pca = pd.DataFrame(reduced)
        print(np.linalg.matrix_rank(self.skeleton_pca))
        self.kmeans(self.skeleton_pca, plot_type="3d",n_clusters=3)
        #self.n_clustering_number()

    def plot_2d(self, n_clusters, km, clusters, df):
        plt.figure(num=None, figsize=(10, 10), dpi=100,
                   facecolor='w', edgecolor='k')

        for i in range(n_clusters):

            print("The centroid of the cluster number " + str(i) +
                  " is " + str(clusters[i, 0]) + " " + str(clusters[i, 1]))
            plt.scatter(df[km.labels_ == i][0], df[km.labels_ == i][1],
                        alpha=0.25, s=100, cmap='rainbow')
            plt.scatter(clusters[i, 0], clusters[i, 1],
                        s=100, alpha=0.30, color="black")
            # find the frame with the minimum distance to the centroid
            min_dista = 1000000
            min_frame = 0
            for j in range(df[km.labels_ == i][0].shape[0]):
                dista = np.sqrt((df[km.labels_ == i][0].iloc[j] - clusters[i, 0])**2 + (df[km.labels_ == i]
                                [1].iloc[j] - clusters[i, 1])**2)
                if dista < min_dista:
                    min_dista = dista
                    min_frame = j
            print("The frame with the minimum distance to the centroid is " + str(min_frame) + " so the frame is in the minute " +
                  str(round(min_frame/60)) + " and second " + str(min_frame % 60) + " of the video")
        plt.show()

    def plot_3d(self, n_clusters, km, clusters, df):
        plt.figure(num=None, figsize=(10, 10), dpi=100,
                   facecolor='w', edgecolor='k')

        ax = plt.subplot(111, projection='3d')
        # another plot but 2d
        for i in range(n_clusters):
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

    def kmeans(self, df, n_clusters=30, plot_type="2d", save_video=True):
        print()
        print("Kmeans")
        # apply kmeans with 6 clusters
        km = KMeans(n_clusters=n_clusters, n_init="auto").fit(df)
        # get the centroids of each cluster
        clusters = km.cluster_centers_
        # plot the data
        self.plot_2d(n_clusters, km, clusters, df) if plot_type == "2d" else self.plot_3d(
            n_clusters, km, clusters, df)
        
        os.makedirs("Videos", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, layers = self.video[1].shape
        # Codec (e.g., 'XVID', 'MJPG', 'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10.0
        video = [cv2.VideoWriter(f"Videos/cluster{i}.mp4", fourcc, fps, (width, height))
                 for i in range(max(km.labels_)+1)]
        if save_video:
            print("Saving Videos")
            for k, image in enumerate(self.video):
                if k > len(km.labels_)-1:
                    break
                label = km.labels_[k]
                video[label].write(image)
            for v in video:
                v.release()

    def t_SNE(self, df,n_clusters):
        print()
        print("t-SNE")
        # calculate the number of components of the t-SNE algorithm
        tsne = TSNE(n_components=2, perplexity=50,
                    verbose=2).fit_transform(df)
        tsne = pd.DataFrame(tsne)
        self.kmeans(tsne, plot_type="2d",n_clusters=n_clusters)
        #self.n_clustering_number(tsne)

    def load_video(self, video_location):
        print("\nLoading Video")
        # Open the video fie
        video = cv2.VideoCapture(video_location)
        self.video = []

        # Check if the video was successfully opened
        if not video.isOpened():
            print('Error opening video file')
            return
        i=0
        # Read and Save frames from the video
        while True:
            # Read a frame from the video
            ret, frame = video.read()

            if not ret:
                break
            if i!=0:
            # Save the frame
                self.video.append(frame)
            i=1

        video.release()
        cv2.destroyAllWindows()
    def n_clustering_number(self,df):
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(1,20))
        visualizer.fit(df)
        visualizer.show()
        plt.show()
# %%


def main():
    data = BData("Data/girosmallveryslow2_openpose.mat",
                 "Data/girosmallveryslow2_vggfeatures.mat",
                 "Data/girosmallveryslow2.mp4")
    parser = argparse.ArgumentParser(description='Argument parser')

    # Add the "-vgg" option
    parser.add_argument('-vgg', choices=['pca', 't-sne'], help='VGG option')

    # Add the "skeleton" option with choices for PCA and t-sne
    parser.add_argument('-skeleton', choices=['pca', 't-sne'], help='Openpose option')

    parser.add_argument('mat', help='Mat location')
    parser.add_argument('video', help='Video location')
    # Parse the command-line arguments
    args = parser.parse_args()
    data.load_video(args.video)
    # VGG Options
    if args.vgg:
        data.VGG = np.array(scipy.io.loadmat(
            args.mat)["features"]).transpose()
        data.EDA_VGG()
        data.center_data(data.VGG_df)

        if args.vgg=='pca':
            data.PCA_VGG()
        elif args.vgg=='t-sne':
            data.t_SNE(data.VGG_df,4)

    #Skeleton Options
    if args.skeleton:
        data.skeleton = np.array(
            scipy.io.loadmat(args.mat)["skeldata"])[:,1:].transpose()
        data.EDA_skelly()
        data.missing_data()
        data.data_manipulation()
        if args.skeleton == 'pca':
            data.PCA_skelly()
        elif args.skeleton == 't-sne':
            data.t_SNE(data.skeleton_df,4)
        
        


if __name__ == '__main__':
    main()
