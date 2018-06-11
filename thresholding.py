import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100), stack=False):
    img = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    if stack:
        # Stack each channel
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        return combined_binary, color_binary
    else:
        return combined_binary, None


def main():
    images = glob.glob('test_images/test*.jpg')

    for fname in images:
        print("Processing", fname)
        img = cv2.imread(fname)
        combined_binary, color_binary = threshold(img, stack=True)

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(32, 9))
        ax1.set_title('Stacked thresholds', fontsize=20)
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax2.set_title('Combined S channel and gradient thresholds', fontsize=20)
        ax2.imshow(combined_binary, cmap='gray')

        ax3.imshow(color_binary)
        ax3.set_title('Color stacked', fontsize=20)

        plt.savefig('output_images/threshold_' + fname.split('/')[-1])


if __name__ == '__main__':
    main()

