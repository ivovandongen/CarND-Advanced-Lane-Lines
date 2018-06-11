import cv2
import numpy as np


class PerspectiveTransform:

    def __init__(self, src, dst):
        self.M = cv2.getPerspectiveTransform(src, dst)

    def transform(self, image, img_size=None):
        return cv2.warpPerspective(image, self.M,
                                   img_size if img_size is not None else (image.shape[1], image.shape[0]),
                                   flags=cv2.INTER_LINEAR)


def main():

    print("Preparing image")
    image = cv2.imread('test_images/straight_lines1.jpg')
    img_shape = image.shape
    height = img_shape[0]
    width = img_shape[1]

    poly_height = int(height / 2 * 1.2)
    polygon = [[180, height], [width // 2 - 100, poly_height], [width // 2 + 100, poly_height], [width - 180, height]]

    image = cv2.fillPoly(image, np.array([polygon]), color=255)

    print("Calculating perspective transform matrix")
    dst = [[180, height], [180, 0], [width - 180, 0], [width - 180, height]]
    transform = PerspectiveTransform(np.float32(polygon), np.float32(dst))

    print("Showing warped image")
    cv2.imshow('img', transform.transform(image))

    cv2.waitKey(5000)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
