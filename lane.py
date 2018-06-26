from lane_line import LaneLine
from line_locator import find_lines_with_sliding_windows, find_lines_from_previous
import cv2
import numpy as np
from collections import deque
from itertools import islice
from glob import glob

from camera_calibration import CameraCalibration
from transform import PerspectiveTransform, Transform
from thresholding import threshold


def _draw_lane_overlay(img, left_fitx, right_fitx, transform, draw_lines=False,
                       fill_lane=True):
    """
    Draws the lane polygon and or lines on the provided image
    :param img: the input image to overlay on
    :param left_fitx: the fitted x coordinates of the left lane line on a linear y-plane of the image after transform
    :param right_fitx: same for the right lane
    :param transform: the transform to convert the lane back onto the image
    :param draw_lines: True if lane lines need to be drawn
    :param fill_lane: True if the lane fill needs to be drawn
    :return: the image with the requested overlays
    """
    height = int(img.shape[0] * transform.y_scale())
    y_points = np.linspace(0, height - 1, height)
    overlay = np.zeros((height, img.shape[1], img.shape[2])).astype(np.uint8)

    # Create lists of points for both lines
    pts_left = np.transpose(np.vstack([left_fitx, y_points]))
    pts_right = np.transpose(np.vstack([right_fitx, y_points]))

    if fill_lane:
        # Make a polygon out of the points of both lines
        pts = np.hstack((np.array([pts_left]), np.array([np.flipud(pts_right)])))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay, np.int_([pts]), (0, 255, 0))
        overlay = transform.transform(overlay)
        img = cv2.addWeighted(img, 1, overlay, 0.3, 0)

    if draw_lines:
        # Draw lines on top
        overlay = np.zeros_like(img).astype(np.uint8)
        cv2.polylines(overlay, np.int32([pts_left, pts_right]), False, (255, 0, 0), thickness=15)
        overlay = transform.transform(overlay)
        img[(overlay > 0)] = 255

    return img


class DetectionFrame:
    """
    Represents line detections in a single frame
    """

    def __init__(self, left: LaneLine, right: LaneLine):
        self.left_lane = left
        self.right_lane = right
        if left.is_valid() and right.is_valid():
            # print(self.left_lane.curvature_m, self.right_lane.curvature_m)
            self.curvature_m = (self.left_lane.curvature_m + self.right_lane.curvature_m) / 2
            self.lane_width_start_m = self.right_lane.offset_start_m - self.left_lane.offset_start_m
            self.offset_m = -(self.left_lane.offset_start_m + self.right_lane.offset_start_m)

    def is_valid(self):
        return self.left_lane.is_valid() \
               and self.right_lane.is_valid() \
               and self.lane_width_valid() \
               and self.lanes_are_parallel()
               # and self.lane_line_curvatures_valid() \

    def lane_line_curvatures_valid(self, margin=10):
        # TODO express in percentage
        difference = abs(self.left_lane.curvature_m - self.right_lane.curvature_m)
        percentage = abs(difference / self.left_lane.curvature_m * 100)
        return percentage < margin

    def lane_width_valid(self, ideal_lane_width=3.7, margin=.5):
        return abs(self.lane_width_start_m - ideal_lane_width) < margin

    def lanes_are_parallel(self, max_difference=50):
        last = len(self.left_lane.fit_x) // 2
        difference = abs(abs(self.left_lane.fit_x[0] - self.left_lane.fit_x[last]) - abs(self.right_lane.fit_x[0] - self.right_lane.fit_x[last]))
        return difference < max_difference


class Lane:
    """
    Class to track lane information over multiple frames of video
    """

    def __init__(self):
        self.history = deque(maxlen=30)
        self.invalid_counter = 0

    def previous_frame(self):
        return self.history[0] if len(self.history) > 0 else None

    def _detect_frame(self, warped_image):
        # First try with a previous frame
        previous_frame = self.previous_frame()
        if previous_frame is not None:
            left_lane, right_lane = find_lines_from_previous(warped_image,
                                                             previous_frame.left_lane,
                                                             previous_frame.right_lane)
            frame = DetectionFrame(left_lane, right_lane)
            if frame.is_valid():
                return frame

        print("Initial detection")
        left_lane, right_lane = find_lines_with_sliding_windows(warped_image)
        frame = DetectionFrame(left_lane, right_lane)
        if len(self.history) == 0 or frame.is_valid():
            return frame

        return None

    def _curve_radius_average(self):
        """
        Calculates the average radius over the available frame history
        :return: the radius (float)
        """
        curves = np.array([frame.curvature_m for frame in self.history if frame is not None])
        curves = curves[abs(curves - np.mean(curves)) < 1 * np.std(curves)] if len(curves) > 1 else curves
        return np.mean(curves) if len(curves) > 0 else -1

    def _lane_lines_average(self):
        """
        Averages the lane lines over (a part of) the available frame history
        :return: the fitted x coordinates for left, right (or None, None if history is not available)
        """
        if len(self.history) == 0:
            return None, None
        elif len(self.history) == 1:
            return self.history[0].left_lane.fit_x, self.history[0].right_lane.fit_x
        # else:
        #     return self.history[0].left_lane.fit_x, self.history[0].right_lane.fit_x
        else:
            end_index = min(len(self.history), 5)
            weights = [100, 10] if end_index == 2 else [100, 10, 1]
            left_poly = np.average([frame.left_lane.poly for frame in islice(self.history, end_index)], axis=0)
            right_poly = np.average([frame.right_lane.poly for frame in islice(self.history, end_index)], axis=0,)
            # weights = [100, 10] if end_index == 2 else [100, 10, 1]
            # left_poly = np.average([frame.left_lane.poly for frame in islice(self.history, end_index)], axis=0,
            #                         weights=weights)
            # right_poly = np.average([frame.right_lane.poly for frame in islice(self.history, end_index)], axis=0,
            #                         weights=weights)
            fit_y = self.history[0].left_lane.fit_y
            return LaneLine.calculate_points_along_line(left_poly, fit_y), LaneLine.calculate_points_along_line(
                right_poly, fit_y)
            # end_index = min(len(self.history), 3)
            # weights = [100, 10] if end_index == 2 else [100, 10, 1]
            # left_poly = np.average([frame.left_lane.poly for frame in islice(self.history, end_index)], axis=0, weights=weights)
            # right_poly = np.average([frame.right_lane.poly for frame in islice(self.history, end_index)], axis=0, weights=weights)
            # fit_y = self.history[0].left_lane.fit_y
            # return LaneLine.calculate_points_along_line(left_poly, fit_y), LaneLine.calculate_points_along_line(right_poly, fit_y)

    def update(self, warped_image):
        """
        Call to update the lane with the most recent frame or still image
        :param warped_image: the warped binary image representing the video frame or still image
        :return: True a detection could be made
        """
        frame = self._detect_frame(warped_image)

        if frame is not None:
            self.history.appendleft(frame)
            self.invalid_counter = 0
            return True
        else:
            self.invalid_counter += 1
            print("Invalid frames", self.invalid_counter)
            return False

    def overlay(self, image, transform: Transform = Transform(), draw_lines=False, fill_lane=True):
        # Ensure the polyfit could be calculated
        avg_left, avg_right = self._lane_lines_average()
        if avg_left is not None and avg_right is not None:
            return _draw_lane_overlay(image,
                                      avg_left,
                                      avg_right,
                                      transform=transform,
                                      draw_lines=draw_lines,
                                      fill_lane=fill_lane)
        else:
            print("skipping")
            return image

    def overlay_text(self, image, anchor_point=(10, 500), font_scale=1, font_color=(255, 255, 255),
                     line_type=2, font=cv2.FONT_HERSHEY_COMPLEX):
        previous_frame = self.previous_frame()
        if previous_frame is not None:
            radius = self._curve_radius_average()
            entries = list()
            entries.append(("Radius:", "{: >10.2f}m".format(radius) if radius < 5000 else "    straight"))
            entries.append(("Offset:", "{: >10.2f}m".format(previous_frame.offset_m)))  # todo: average?
            entries.append(("Lane width:", "{: >10.2f}m".format(previous_frame.lane_width_start_m)))  # todo average?

            for i, l in enumerate(entries):
                y1 = anchor_point[1] + (i * 70)
                y2 = y1 + 30
                x1 = anchor_point[0]
                x2 = anchor_point[0] + 50
                image = cv2.putText(image, l[0], (x1, y1), font, font_scale, font_color, line_type)
                image = cv2.putText(image, l[1], (x2, y2), font, font_scale, font_color, line_type)

        return image


def main():
    # Fixed image dimensions
    height = 720
    width = 1280

    # Prepare camera calibration
    print("Calibrating camera")
    calibration = CameraCalibration.default()
    transform = PerspectiveTransform.default(height, width)

    images = glob('test_images/straight*') + glob('test_images/test*')

    for fname in images:
        print("Processing", fname)
        org_image = cv2.imread(fname)
        if org_image.shape != (height, width, 3):
            print("skipping image", fname, "invalid dimensions", org_image.shape)
            continue

        # Run the pipeline on a copy of the image
        undistorted = calibration.undistort(np.copy(org_image))
        transformed = transform.transform(undistorted)
        warped_binary, _ = threshold(transformed, stack=False)
        # combined, _ = threshold(undistorted, stack=False)
        # warped_binary = transform.transform(combined)

        lane = Lane()
        lane.update(warped_binary)
        final = lane.overlay(org_image, draw_lines=False, fill_lane=True, transform=transform.invert())
        final = lane.overlay_text(final)
        final = cv2.polylines(final, [np.int32(transform.src)], color=[255, 0, 0], isClosed=False)

        cv2.imwrite('output_images/lane_{}'.format(fname.split('/')[-1]), final)

        final = lane.overlay(np.dstack((warped_binary, warped_binary, warped_binary)) * 255, draw_lines=True,
                             fill_lane=False)
        cv2.imwrite('output_images/lane_warped_{}'.format(fname.split('/')[-1]), final)


if __name__ == '__main__':
    main()
