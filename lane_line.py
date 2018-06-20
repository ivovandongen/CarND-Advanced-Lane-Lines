import math

import numpy as np


class LaneLine():
    def __init__(self, poly, indices, image_shape, detection_windows=None):
        self.poly = poly
        self.indices = indices
        self.detection_windows = detection_windows
        self.image_shape = image_shape
        # # was the line detected in the last iteration?
        # self.detected = False
        # # x values of the last n fits of the line
        # self.recent_xfitted = []
        # # average x values of the fitted line over the last n iterations
        # self.bestx = None
        # # polynomial coefficients averaged over the last n iterations
        # self.best_fit = None
        # # polynomial coefficients for the most recent fit
        # self.current_fit = [np.array([False])]
        # # radius of curvature of the line in some units
        # self.radius_of_curvature = None
        # # distance in meters of vehicle center from the line
        # self.line_base_pos = None
        # # difference in fit coefficients between last and new fits
        # self.diffs = np.array([0, 0, 0], dtype='float')
        # # x values for detected line pixels
        # self.allx = None
        # # y values for detected line pixels
        # self.ally = None

    def is_valid(self):
        return len(self.poly) >= 3

    def calculate_points_along_line(self, y_points):
        return self.poly[0] * y_points ** 2 + self.poly[1] * y_points + self.poly[2]

    def calculate_curvature_m(self):
        image_scale = (self.image_shape[0] / 720)
        yscale = 30 / 720 / image_scale  # Real world metres per y pixel
        xscale = 3.7 / 700  # Real world metres per x pixel

        # Convert polynomial to set of points for refitting
        ploty = np.linspace(0, self.image_shape[0] * 3 - 1, self.image_shape[0])
        fitx = self.poly[0] * ploty ** 2 + self.poly[1] * ploty + self.poly[2]

        # Fit new polynomial
        fit_cr = np.polyfit(ploty * yscale, fitx * xscale, 2)

        # Calculate curve radius
        return ((1 + (2 * fit_cr[0] * np.max(ploty) * yscale + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])