import numpy as np
import cv2
import matplotlib.pyplot as plt
from lane_line import LaneLine


def find_lines_from_previous(binary_warped, left_line: LaneLine, right_lane: LaneLine):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_fit = left_line.poly
    right_fit = right_lane.poly

    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return LaneLine(left_fit, left_lane_inds, binary_warped.shape), \
           LaneLine(right_fit, right_lane_inds, binary_warped.shape)


def find_lines_with_sliding_windows(binary_warped, num_windows=9, window_margin=100, pixel_threshold=50):
    """

    :param binary_warped: the input image
    :param num_windows: the number of windows per line to use
    :param window_margin: the width of the windows +/- margin
    :param pixel_threshold: the minimum number of pixels found to recenter window
    :return: left_lane: LaneLine, right_lane: LaneLine
    """

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] // num_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Create lists to store the search windows
    left_lane_windows = []
    right_lane_windows = []

    # Step through the windows one by one
    for window in range(num_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - window_margin
        win_xleft_high = leftx_current + window_margin
        win_xright_low = rightx_current - window_margin
        win_xright_high = rightx_current + window_margin

        # Collect the windows
        left_lane_windows.append(((win_xleft_low, win_y_low), (win_xleft_high, win_y_high)))
        right_lane_windows.append(((win_xright_low, win_y_low), (win_xright_high, win_y_high)))

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > pixel_threshold:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > pixel_threshold:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2) if len(lefty) > 0 and len(leftx) > 0 else [np.array([False])]
    right_fit = np.polyfit(righty, rightx, 2) if len(righty) > 0 and len(rightx) > 0 else [np.array([False])]

    return LaneLine(left_fit, left_lane_inds, binary_warped.shape, left_lane_windows), \
           LaneLine(right_fit, right_lane_inds, binary_warped.shape, right_lane_windows)


def plot_sliding_windows(binary_warped, left_lane: LaneLine, right_lane: LaneLine):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_lane.fit_x
    right_fitx = right_lane.fit_x

    # Create an output image to draw on and  visualize the result
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    out_img[nonzeroy[left_lane.indices], nonzerox[left_lane.indices]] = [255, 0, 0]
    out_img[nonzeroy[right_lane.indices], nonzerox[right_lane.indices]] = [0, 0, 255]

    for left_win, right_win in zip(left_lane.detection_windows, right_lane.detection_windows):
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, left_win[0], left_win[1], (0, 255, 0), 2)
        cv2.rectangle(out_img, right_win[0], right_win[1], (0, 255, 0), 2)

    plt.clf()
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)


def main():
    from glob import glob
    from camera_calibration import CameraCalibration
    from thresholding import threshold
    from transform import PerspectiveTransform

    # Fixed image dimensions
    height = 720
    width = 1280

    # Prepare camera calibration
    print("Calibrating camera")
    calibration = CameraCalibration.default()

    # Prepare perspective transform
    transform = PerspectiveTransform.default(height, width)

    images = glob('test_images/straight*') + glob('test_images/test*')

    for fname in images:
        print("Processing", fname)

        # Run pipeline
        img = cv2.imread(fname)
        img = calibration.undistort(img)
        img, _ = threshold(img)
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (5,5))
        img = transform.transform(img)

        # Find lines using sliding windows
        left_lane, right_lane = find_lines_with_sliding_windows(img)

        # Plot sliding windows
        plot_sliding_windows(img, left_lane, right_lane)

        # combined_binary, color_binary = threshold(img, stack=True)
        #
        # f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(32, 9))
        # ax1.set_title('Stacked thresholds', fontsize=20)
        # ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #
        # ax2.set_title('Result', fontsize=20)
        # ax2.imshow(combined_binary, cmap='gray')

        plt.savefig('output_images/sliding_windows_' + fname.split('/')[-1])


if __name__ == '__main__':
    main()
