from lane_line import LaneLine
from line_locator import find_lines_with_sliding_windows
import cv2
import numpy as np

from camera_calibration import default_camera_calibration
from transform import PerspectiveTransform
from thresholding import threshold


def _draw_lane_overlay(img, left_fit, right_fit, transform_fn=None, draw_lines=False, fill_lane=True):
    height = img.shape[0]
    ploty = np.linspace(0, height - 1, height)
    overlay = np.zeros_like(img).astype(np.uint8)

    # Calculate points.
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create lists of points for both lines
    pts_left = np.transpose(np.vstack([left_fitx, ploty]))
    pts_right = np.transpose(np.vstack([right_fitx, ploty]))

    if fill_lane:
        # Make a polygon out of the points if both lines
        pts = np.hstack((np.array([pts_left]), np.array([np.flipud(pts_right)])))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay, np.int_([pts]), (0, 255, 0))
        overlay = overlay if transform_fn is None else transform_fn(overlay)
        img = cv2.addWeighted(img, 1, overlay, 0.3, 0)

    if draw_lines:
        # Draw lines on top
        overlay = np.zeros_like(img).astype(np.uint8)
        cv2.polylines(overlay, np.int32([pts_left, pts_right]), False, (255, 0, 0), thickness=15)
        overlay = overlay if transform_fn is None else transform_fn(overlay)
        img[(overlay > 0)] = 255

    return img


class Lane:
    """
    Class to track lane information over multiple frames of video
    """

    def __init__(self):
        self.left_lane = LaneLine()
        self.right_lane = LaneLine()
        self.invalid_frames = 0
        self.result = None

    def update(self, warped_image):
        self.result = find_lines_with_sliding_windows(warped_image)
        return True

    def overlay(self, image, transform_fn=None, draw_lines=False, fill_lane=True):
        # Ensure the polyfit could be calculated
        (left_fit, right_fit, left_lane_inds, right_lane_inds, left_lane_windows, right_lane_windows) = self.result
        if len(left_fit) >= 3 and len(right_fit) >= 3:
            return _draw_lane_overlay(image, left_fit, right_fit,
                                      transform_fn=transform_fn,
                                      draw_lines=draw_lines,
                                      fill_lane=fill_lane)
        else:
            print("skipping")
            return image

    def isValid(self):
        return False


def main():
    # Fixed image dimensions
    height = 720
    width = 1280

    # Prepare camera calibration
    print("Calibrating camera")
    calibration = default_camera_calibration()

    # Prepare perspective transform
    poly_height = int(height * .35)
    bottom_offset = 80
    top_offset = 120
    polygon = [[bottom_offset, height], [width // 2 - top_offset, height - poly_height],
               [width // 2 + top_offset, height - poly_height], [width - bottom_offset, height]]

    print("Calculating perspective transform matrix")
    dst = [[bottom_offset, height], [bottom_offset, 0], [width - bottom_offset, 0], [width - bottom_offset, height]]
    transform = PerspectiveTransform(np.float32(polygon), np.float32(dst))

    org_image = cv2.imread("test_images/test1.jpg")

    # Run the pipeline on a copy of the image
    image = calibration.undistort(np.copy(org_image))
    # image = transform.transform(image)
    combined, _ = threshold(image, stack=False)
    warped_binary = transform.transform(combined)

    lane = Lane()
    lane.update(warped_binary)
    final = lane.overlay(org_image, draw_lines=True, fill_lane=False, transform_fn=lambda x: transform.inverse(x))

    cv2.imwrite('output_images/lane_test1.jpg', final)


if __name__ == '__main__':
    main()
