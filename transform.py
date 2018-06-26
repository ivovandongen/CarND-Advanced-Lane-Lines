import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


class Transform:

    def transform(self, image):
        return image

    def invert(self):
        return self

    def x_scale(self):
        return 1

    def y_scale(self):
        return 1


class PerspectiveTransform(Transform):
    """
    Manages bi-directional perspective transformations
    """

    def __init__(self, src, dst, src_size, dst_size, M=None, Minv=None):
        self.src = src
        self.src_size = src_size
        self.dst = dst
        self.dst_size = dst_size
        self.M = M if M is not None else cv2.getPerspectiveTransform(src, dst)
        self.Minv = Minv if Minv is not None else cv2.getPerspectiveTransform(dst, src)

    def transform(self, image):
        return cv2.warpPerspective(image, self.M,
                                   (self.dst_size[1], self.dst_size[0]),
                                   flags=cv2.INTER_LINEAR)

    def invert(self):
        return PerspectiveTransform(src=self.dst, dst=self.src,
                                    src_size=self.dst_size, dst_size=self.src_size,
                                    M=self.Minv, Minv=self.M)

    def x_scale(self):
        return self.src_size[1] / self.dst_size[1]

    def y_scale(self):
        return self.src_size[0] / self.dst_size[0]

    @staticmethod
    def default(height=720, width=1280):
        print("Calculating perspective transform matrix")

        poly_height = int(height * .35)  # int(height * .35)
        bottom_offset_left = 60
        bottom_offset_right = bottom_offset_left
        bottom_margin = 0
        top_offset = 120
        polygon = [[bottom_offset_left, height - bottom_margin],
                   [width // 2 - top_offset, height - poly_height],
                   [width // 2 + top_offset, height - poly_height],
                   [width - bottom_offset_right, height - bottom_margin]]

        margin_x_bottom = 200
        margin_x_top = 100
        dst_height = height #* 3
        dst_width = width
        dst = [[margin_x_bottom, dst_height],
               [margin_x_top, 0],
               [dst_width - margin_x_top, 0],
               [dst_width - margin_x_bottom, dst_height]]

        return PerspectiveTransform(np.float32(polygon), np.float32(dst), src_size=(height, width),
                                    dst_size=(dst_height, dst_width))


def main():
    # # Fixed image dimensions
    height = 720
    width = 1280
    transform = PerspectiveTransform.default(height, width)
    from camera_calibration import CameraCalibration
    calibration = CameraCalibration.default()

    images = glob('test_images/straight*') + glob('test_images/test*')

    for fname in images:
        print("Processing", fname)
        image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        image = calibration.undistort(image)

        polygonned = cv2.polylines(np.copy(image), [np.int32(transform.src)], False, color=255, thickness=1)

        transformed = transform.transform(np.copy(polygonned))

        inversed = transform.invert().transform(np.copy(transformed))

        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(32, 9))
        ax1.set_title('Original', fontsize=20)
        ax1.imshow(image)

        ax2.set_title('Area of interest', fontsize=20)
        ax2.imshow(polygonned)

        ax3.set_title('Transformed', fontsize=20)
        ax3.imshow(transformed)

        ax4.set_title('Transformed inversed', fontsize=20)
        ax4.imshow(inversed)

        plt.savefig('output_images/transform_' + fname.split('/')[-1])


if __name__ == '__main__':
    main()
