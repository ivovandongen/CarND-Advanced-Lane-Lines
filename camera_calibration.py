import numpy as np
import cv2
import glob


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

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
        if ret:
            self.matrix = mtx
            self.distortion = dist

    def undistort(self, img):
        assert (self.matrix is not None)
        return cv2.undistort(img, self.matrix, self.distortion, None, self.matrix)


def main():
    # Make a list of calibration images
    print("Calculating distortion matrix")
    images = glob.glob('camera_cal/calibration*.jpg')
    calibration = CameraCalibration()
    calibration.calibrate(images, checkerboard_shape=(9, 6), img_shape=(1280, 720))

    print("Showing undistorted images")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        undist = calibration.undistort(img)
        cv2.imshow('img', undist)
        cv2.waitKey(500)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
