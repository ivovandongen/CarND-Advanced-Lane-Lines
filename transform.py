import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


class PerspectiveTransform:

    def __init__(self, src, dst):
        self.M = cv2.getPerspectiveTransform(src, dst)

    def transform(self, image, img_size=None):
        return cv2.warpPerspective(image, self.M,
                                   img_size if img_size is not None else (image.shape[1], image.shape[0]),
                                   flags=cv2.INTER_LINEAR)


def main():

    # Fixed image dimensions
    height = 720
    width = 1280

    poly_height = int(height * .35)
    bottom_offset = 80
    top_offset = 120
    polygon = [[bottom_offset, height], [width // 2 - top_offset, height - poly_height], [width // 2 + top_offset, height - poly_height], [width - bottom_offset, height]]

    print("Calculating perspective transform matrix")
    dst = [[bottom_offset, height], [bottom_offset, 0], [width - bottom_offset, 0], [width - bottom_offset, height]]
    transform = PerspectiveTransform(np.float32(polygon), np.float32(dst))

    images = glob('test_images/test*')

    for fname in images:
        print("Processing", fname)
        image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)

        polygonned = cv2.polylines(np.copy(image), np.array([polygon]), False, color=255, thickness=2)

        transformed = transform.transform(np.copy(polygonned), (width, height))

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(32, 9))
        ax1.set_title('Original', fontsize=20)
        ax1.imshow(image)

        ax2.set_title('Area of interest', fontsize=20)
        ax2.imshow(polygonned)

        ax3.set_title('Transformed', fontsize=20)
        ax3.imshow(transformed)

        plt.savefig('output_images/transform_' + fname.split('/')[-1])


if __name__ == '__main__':
    main()
