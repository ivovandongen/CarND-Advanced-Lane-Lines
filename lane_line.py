import numpy as np


class LaneLine():
    def __init__(self, poly, indices, image_shape, detection_windows=None):
        self.poly = poly
        self.indices = indices
        self.detection_windows = detection_windows
        self.image_shape = image_shape
        self.image_scale = self.image_shape[0] / 720
        self.yscale = 30 / 720 / self.image_scale  # Real world metres per y pixel
        self.xscale = 3.7 / 700  # Real world metres per x pixel
        self.fit_y = np.linspace(0, self.image_shape[0] - 1, self.image_shape[0])
        self.fit_x = self.calculate_points_along_line(self.poly, self.fit_y)
        self.offset_start_m = self.calculate_offset_start_m()
        self.poly_scaled = self.fit_scaled()
        self.curvature_m = self.calculate_curvature_m()

    def calculate_curvature_m(self):
        return ((1 + (2 * self.poly_scaled[0] * np.max(self.fit_y) * self.yscale + self.poly_scaled[
            1]) ** 2) ** 1.5) / np.absolute(2 * self.poly_scaled[0])

    def fit_scaled(self):
        return np.polyfit(self.fit_y * self.yscale, self.fit_x * self.xscale, 2)

    def calculate_offset_start_m(self):
        return (self.calculate_points_along_line(self.poly, self.image_shape[0]) - self.image_shape[1] // 2) * self.xscale

    @staticmethod
    def calculate_points_along_line(poly, y_points):
        return poly[0] * y_points ** 2 + poly[1] * y_points + poly[2]

    def is_valid(self):
        return len(self.poly) >= 3
