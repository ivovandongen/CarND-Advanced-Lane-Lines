import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from camera_calibration import default_camera_calibration
from transform import PerspectiveTransform
from thresholding import threshold
from lane import Lane


class VideoProcessor:
    """
    Class to help with processing a video
    """

    def __init__(self, input_file, output_file, image_size):
        """
        Creates the processor
        :param input_file: the input video file name
        :param output_file: the output video file name
        :param image_size: the iamge size of the video
        """
        self.input_file = input_file
        self.output_file = output_file
        self.lane = Lane()
        self.image_size = image_size
        self.calibration = self.create_camera_calibration()
        self.transform = self.create_transform(self.image_size)

    @staticmethod
    def create_camera_calibration():
        return default_camera_calibration()

    @staticmethod
    def create_transform(image_size):
        height = image_size[0]
        width = image_size[1]

        # Prepare perspective transform
        poly_height = int(height * .35)
        bottom_offset = 80
        bottom_margin = 40
        top_offset = 120
        polygon = [[bottom_offset, height - bottom_margin], [width // 2 - top_offset, height - poly_height],
                   [width // 2 + top_offset, height - poly_height], [width - bottom_offset, height - bottom_margin]]

        print("Calculating perspective transform matrix")
        dst = [[bottom_offset, height], [bottom_offset, 0], [width - bottom_offset, 0], [width - bottom_offset, height]]
        return PerspectiveTransform(np.float32(polygon), np.float32(dst))

    def process(self, sub_clip=None):
        clip = VideoFileClip(self.input_file)
        if sub_clip:
            clip = clip.subclip(sub_clip[0], sub_clip[1])
        out_clip = clip.fl_image(lambda image: self.process_image(image))
        out_clip.write_videofile(self.output_file, audio=False)

    def create_birds_eye_view(self, warped_image, scale_factor=3):
        """
        Create a image of the birds-eye view projection
        :param warped_image: warped binary image
        :param scale_factor: down-size scale factor
        :return: RGB scaled image
        """

        # Stack the binary image and expand the color range
        warped_image = np.dstack((warped_image, warped_image, warped_image)) * 255
        overlay_large = self.lane.overlay(warped_image, draw_lines=True, fill_lane=False)

        # Resize
        return cv2.resize(overlay_large,
                          dsize=(self.image_size[1] // scale_factor, self.image_size[0] // scale_factor),
                          interpolation=cv2.INTER_CUBIC)

    def process_image(self, org_image):
        # Create the warped binary image (birds-eye view with lane lines highlighted)
        image = self.calibration.undistort(np.copy(org_image))
        combined, _ = threshold(image, stack=False)
        warped = self.transform.transform(combined)

        # Update our lane tracker
        self.lane.update(warped_image=warped)

        # Add the lane overlay to our original image
        org_image = self.lane.overlay(org_image, transform_fn=lambda x: self.transform.inverse(x))

        # Create picture-in-picture overlays
        birds_eye_overlay = self.create_birds_eye_view(warped_image=warped)

        # Overlay result picture-in-picture style on original
        y_offset, x_offset = 0, 0
        org_image[y_offset:y_offset + birds_eye_overlay.shape[0],
        x_offset:x_offset + birds_eye_overlay.shape[1]] = birds_eye_overlay
        # x_offset = org_image.shape[1] - pipe2_result.shape[1]
        # org_image[y_offset:y_offset + pipe2_result.shape[0], x_offset:x_offset + pipe2_result.shape[1]] = pipe2_result

        return org_image


def main():
    input_file = "test_videos/project_video.mp4"
    output_file = 'output_videos/processed_project_video.mp4'
    processor = VideoProcessor(input_file=input_file, output_file=output_file, image_size=(720, 1280))
    print("Processing video", input_file, output_file)
    processor.process(sub_clip=(38, 42))


if __name__ == '__main__':
    main()
