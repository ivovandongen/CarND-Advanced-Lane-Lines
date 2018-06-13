import numpy as np
import cv2
from glob import glob
import pickle
from os import path
import matplotlib.pyplot as plt


class CameraCalibration:

    def __init__(self):
        self.matrix = None
        self.distortion = None

    def calibrate(self, images, checkerboard_shape=(9, 6), img_shape=None):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((checkerboard_shape[0] * checkerboard_shape[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_shape[0], 0:checkerboard_shape[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Step through the list and search for chessboard corner
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Store the image shape
            if img_shape is None:
                img_shape = gray.shape[::-1]

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_shape, None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
            else:
                print("Checkerboard not found, skipping", fname)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
        if ret:
            self.matrix = mtx
            self.distortion = dist


    def undistort(self, img):
        assert (self.matrix is not None)
        return cv2.undistort(img, self.matrix, self.distortion, None, self.matrix)

    def save_to_file(self, file):
        pickle.dump(self, file)


def load_from(file):
    return pickle.load(file)


def default_camera_calibration(reset=False):
    file = 'camera_calibration.p'
    if not reset and path.exists(file):
        print("Return saved camera calibration")
        with open(file, 'rb') as f:
            return load_from(f)
    else:
        print("Calibrating camera")
        calibration = CameraCalibration()
        calibration.calibrate(glob('camera_cal/calibration*'), checkerboard_shape=(9, 6), img_shape=(1280, 720))

        with open(file, 'wb') as f:
            print("Saving calibration for next run")
            calibration.save_to_file(f)

        return calibration


def main():
    # Make a list of calibration images
    print("Calculating distortion matrix")
    calibration = default_camera_calibration()

    images = glob('camera_cal/calibration*.jpg')
    print("Showing undistorted images")
    for fname in images:
        print("Processing", fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        checker_board = cv2.drawChessboardCorners(img, (9, 6), corners, ret) if ret else img
        undist = calibration.undistort(img)

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(32, 9))
        ax1.set_title('Original', fontsize=20)
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax2.set_title('Checkerboard' + ('' if ret else ' could not be applied'), fontsize=20)
        ax2.imshow(checker_board, cmap='gray')

        ax3.set_title('Undistorted', fontsize=20)
        ax3.imshow(undist)

        plt.savefig('output_images/undistorted_' + fname.split('/')[-1])


if __name__ == '__main__':
    main()
