import cv2
import scipy.io


class BData():
    def __init__(self, skeleton_location, VGG_location):
        self.skeleton = scipy.io.loadmat(skeleton_location)["skeldata"]
        self.VGG = scipy.io.loadmat(VGG_location)["features"]


def main():
    data = BData("Data/bestofgiro2020_skeletons.mat",
                 "Data/bestofgiro2020_vggfeatures.mat")


if __name__ == '__main__':
    main()
