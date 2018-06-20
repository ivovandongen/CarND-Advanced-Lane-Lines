import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def blur(image, kernel_size=15):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255, sobel_kernel=9):
    # Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel) if orient == 'x' \
        else cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return this mask as your binary_output image
    binary_output = sxbinary
    return binary_output


def dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3)):

    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    gradient_direction = np.arctan2(abs_sobely, abs_sobelx)

    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(gradient_direction)
    binary_output[(gradient_direction >= thresh[0]) & (gradient_direction <= thresh[1])] = 1

    # Return this mask as your binary_output image
    return binary_output

def mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 100)):

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


def sobel_combined(img):
    gradx = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)
    grady = abs_sobel_thresh(img, orient='y', thresh_min=20, thresh_max=100)
    dir_binary = dir_threshold(img)
    mag_binary = mag_thresh(img)
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


def color_thresh(img, thresh=(0,255)):
    s_binary = np.zeros_like(img)
    s_binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return s_binary


def threshold(img, s_thresh=(90, 255), sx_thresh=(20, 100), sobel_kernel=9, stack=False):
    img = np.copy(img)

    # TODO:
    # Sobel
    # - Which channel to use? H/L/S/Gray
    # - Blurring? Kernel size?
    # - Kernel size?
    # - x/y/xy?
    #
    # Thresholding:
    # - h/l/s 2 or 3 channels?
    # - Threshold values
    # - Separate thresholds / channels to pick out most common
    #   line colors? (white/yellow)
    #
    # - Is it important to pick up on highly shaded lines?
    #   Or can we cant away with this by using extrapolation
    #   from multiple frames?

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Threshold H channel
    thresh = (15, 100)
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel > thresh[0]) & (h_channel <= thresh[1])] = 1

    # Combine
    combined_binary = np.zeros_like(sxbinary)
    # combined_binary[(h_binary == 1) | (sxbinary == 1)] = 1
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    if stack:
        # Stack each channel
        color_binary = np.dstack((h_binary, sxbinary, s_binary)) * 255

        return combined_binary, color_binary
    else:
        return combined_binary, None


def main():
    images = glob.glob('test_images/test*')

    for fname in images:
        print("Processing", fname)
        img = cv2.imread(fname)

        combined_binary, color_binary = threshold(img, stack=True)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h_channel = hls[:, :, 0]
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        f, (r1, r2, r3, r4, r5) = plt.subplots(5, 4, figsize=(32, 22))

        (ax1, ax2, ax3, ax4) = r1
        ax1.set_title('Original', fontsize=20)
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax2.set_title('H', fontsize=20)
        ax2.imshow(h_channel, cmap='gray')

        ax3.imshow(l_channel)
        ax3.set_title('L', fontsize=20)

        ax4.imshow(s_channel, cmap='gray')
        ax4.set_title('S', fontsize=20)

        (ax1, ax2, ax3, ax4) = r2
        ax1.imshow(dir_threshold(gray), cmap='gray')
        ax1.set_title('Sobel direction (gray)', fontsize=20)

        ax2.imshow(dir_threshold(h_channel), cmap='gray')
        ax2.set_title('Sobel direction (H)', fontsize=20)

        ax3.imshow(dir_threshold(l_channel), cmap='gray')
        ax3.set_title('Sobel direction (L)', fontsize=20)

        ax4.imshow(dir_threshold(s_channel), cmap='gray')
        ax4.set_title('Sobel direction (S)', fontsize=20)

        (ax1, ax2, ax3, ax4) = r3
        ax1.imshow(mag_thresh(gray), cmap='gray')
        ax1.set_title('Sobel magnitude (gray)', fontsize=20)

        ax2.imshow(mag_thresh(h_channel), cmap='gray')
        ax2.set_title('Sobel magnitude (H)', fontsize=20)

        ax3.imshow(mag_thresh(l_channel), cmap='gray')
        ax3.set_title('Sobel magnitude (L)', fontsize=20)

        ax4.imshow(mag_thresh(s_channel), cmap='gray')
        ax4.set_title('Sobel magnitude (S)', fontsize=20)

        (ax1, ax2, ax3, ax4) = r4
        ax1.imshow(sobel_combined(gray), cmap='gray')
        ax1.set_title('Sobel combined (gray)', fontsize=20)

        ax2.imshow(sobel_combined(h_channel), cmap='gray')
        ax2.set_title('Sobel combined (H)', fontsize=20)

        ax3.imshow(sobel_combined(l_channel), cmap='gray')
        ax3.set_title('Sobel combined (L)', fontsize=20)

        ax4.imshow(sobel_combined(s_channel), cmap='gray')
        ax4.set_title('Sobel combined (S)', fontsize=20)

        (ax1, ax2, ax3, ax4) = r5
        threshed, color = threshold(img, sobel_kernel=9, s_thresh=(90, 255))
        ax1.imshow(threshed, cmap='gray')
        ax1.set_title('Combined Sobel/Color', fontsize=20)

        ax2.imshow(color_thresh(h_channel, thresh=(0, 100)), cmap='gray')
        ax2.set_title('Color thresh (H)', fontsize=20)

        ax3.imshow(color_thresh(l_channel, thresh=(170, 255)), cmap='gray')
        ax3.set_title('Color thresh (L)', fontsize=20)

        ax4.imshow(color_thresh(s_channel, thresh=(90, 255)), cmap='gray')
        ax4.set_title('Color thresh (S)', fontsize=20)


        plt.savefig('output_images/threshold_' + fname.split('/')[-1])


if __name__ == '__main__':
    main()

