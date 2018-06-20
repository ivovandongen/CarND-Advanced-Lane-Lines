from lane_line import LaneLine
from line_locator import find_lines_with_sliding_windows, find_lines_from_previous
import cv2
import numpy as np
from collections import deque
from glob import glob
import math

from camera_calibration import CameraCalibration
from transform import PerspectiveTransform, Transform
from thresholding import threshold


def _draw_lane_overlay(img, left_lane: LaneLine, right_lane: LaneLine, transform, draw_lines=False,
                       fill_lane=True):
    height = int(img.shape[0] * transform.y_scale())
    y_points = np.linspace(0, height - 1, height)
    overlay = np.zeros((height, img.shape[1], img.shape[2])).astype(np.uint8)#np.zeros_like(img).astype(np.uint8)

    # Calculate points.
    left_fitx = left_lane.calculate_points_along_line(y_points)
    right_fitx = right_lane.calculate_points_along_line(y_points)

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

    def __init__(self, left: LaneLine, right: LaneLine):
        self.left_lane = left
        self.right_lane = right
        self.curvature_m = None
        self.lane_width_m = None
        self.offset_m = None

    def is_valid(self):
        # TODO
        # - Check distance between
        # - Check if parallel
        # - Check if similar curvature
        return self.left_lane.is_valid() \
               and self.right_lane.is_valid() \
               and (abs(self.calculate_lane_width_m() - 3.7) < .8)

    def calculate_curvature_m(self):
        if self.curvature_m is None:
            self.curvature_m = (self.left_lane.calculate_curvature_m() + self.right_lane.calculate_curvature_m()) / 2

        return self.curvature_m

    def calculate_offset_m(self):
        if self.offset_m is None:
            self.offset_m = -(self.left_lane.offset_m + self.right_lane.offset_m)

        return self.offset_m

    def calculate_lane_width_m(self):
        if self.lane_width_m is None:
            self.lane_width_m = self.right_lane.offset_m - self.left_lane.offset_m

        return self.lane_width_m


class Lane:
    """
    Class to track lane information over multiple frames of video
    """

    def __init__(self):
        self.history = deque(maxlen=60)
        self.last_valid_frame = None

    def curve_radius(self):
        curves = np.array([frame.calculate_curvature_m() for frame in self.history if frame is not None])
        curves = curves[abs(curves - np.mean(curves)) < 1 * np.std(curves)] if len(curves) > 1 else curves
        return np.mean(curves) if len(curves) > 0 else -1

    def update(self, warped_image):
        # First try with a previous frame
        if self.last_valid_frame is not None:
            left_lane, right_lane = find_lines_from_previous(warped_image, self.last_valid_frame.left_lane,
                                                             self.last_valid_frame.right_lane)
            frame = DetectionFrame(left_lane, right_lane)
            if frame.is_valid():
                self.history.append(frame)
                self.last_valid_frame = frame
                return

        print("Initial detection")
        left_lane, right_lane = find_lines_with_sliding_windows(warped_image)
        frame = DetectionFrame(left_lane, right_lane)
        if frame.is_valid():
            self.history.append(frame)
            self.last_valid_frame = frame

        # TODO: fall back on history if possible (or do when drawing)

        # if self.last_valid_frame is None:  # TODO self.last_valid_frame is None #len(self.history) == 0:
        #     print("Initial detection")
        #     left_lane, right_lane = find_lines_with_sliding_windows(warped_image)
        #     frame = DetectionFrame(left_lane, right_lane)
        #     self.history.append(frame)
        #     if frame.is_valid():
        #         self.last_valid_frame = frame
        #     #TODO: Else....
        #     # else:
        #     #     self.last_valid_frame
        # else:
        #     left_lane, right_lane = find_lines_from_previous(warped_image, self.last_valid_frame.left_lane, self.last_valid_frame.right_lane)
        #     frame = DetectionFrame(left_lane, right_lane)
        #     if frame.is_valid():
        #         self.last_valid_frame = frame

        # TODO:
        # - check history of last x frames

    def overlay(self, image, transform:Transform=Transform(), draw_lines=False, fill_lane=True):
        # Ensure the polyfit could be calculated
        if self.last_valid_frame is not None:
            # print(self.last_valid_frame.left_lane.calculate_curvature_m(),
            #       self.last_valid_frame.right_lane.calculate_curvature_m())
            frame = self.last_valid_frame
            return _draw_lane_overlay(image, frame.left_lane, frame.right_lane,
                                      transform=transform,
                                      draw_lines=draw_lines,
                                      fill_lane=fill_lane)
        else:
            print("skipping")
            return image

    def overlay_curvature_offset(self, image, bottomLeftCornerOfText=(10, 500), fontScale=1, fontColor=(255, 255, 255),
                                 lineType=2, font=cv2.FONT_HERSHEY_SIMPLEX, debug=False):
        if self.last_valid_frame is not None:
            if debug:
                text = "Left: {}, right: {}, Avg: ".format(self.last_valid_frame.left_lane.calculate_curvature_m(),
                                                           self.last_valid_frame.right_lane.calculate_curvature_m(),
                                                           self.last_valid_frame.calculate_curvature_m())
            else:
                radius = self.curve_radius()
                offset = self.last_valid_frame.calculate_offset_m()
                lane_width = self.last_valid_frame.calculate_lane_width_m()
                text = "Curve radius: {:6.2f}m".format(radius) if radius < 5000 else "Curve radius is straight"
                text += ". Offset: {:6.2f}m".format(offset)
                text += ". Lane width: {:6.2f}m".format(lane_width)

            return cv2.putText(image,
                               text,
                               bottomLeftCornerOfText,
                               font,
                               fontScale,
                               fontColor,
                               lineType)
        else:
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
        final = lane.overlay_curvature_offset(final)
        final = cv2.polylines(final, [np.int32(transform.src)], color=[255, 0, 0], isClosed=False)

        cv2.imwrite('output_images/lane_{}'.format(fname.split('/')[-1]), final)

        final = lane.overlay(np.dstack((warped_binary, warped_binary, warped_binary)) * 255, draw_lines=True,
                             fill_lane=False)
        cv2.imwrite('output_images/lane_warped_{}'.format(fname.split('/')[-1]), final)


if __name__ == '__main__':
    main()
