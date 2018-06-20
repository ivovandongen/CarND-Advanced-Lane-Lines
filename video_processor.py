import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from camera_calibration import CameraCalibration
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
        :param image_size: the image size of the video

        Use VideoProcessor:process to start processing
        """
        self.input_file = input_file
        self.output_file = output_file
        self.lane = Lane()
        self.image_size = image_size
        self.calibration = self.create_camera_calibration()
        self.transform = self.create_transform(self.image_size)

    @staticmethod
    def create_camera_calibration():
        return CameraCalibration.default()

    @staticmethod
    def create_transform(image_size):
        height = image_size[0]
        width = image_size[1]
        return PerspectiveTransform.default(height, width)

    def process(self, sub_clip=None):
        """
        Process the video clip
        :param sub_clip: optionally specify a sub clip (start, end)
        :return: None
        """
        clip = VideoFileClip(self.input_file)
        if sub_clip:
            clip = clip.subclip(sub_clip[0], sub_clip[1])
        out_clip = clip.fl_image(lambda image: self._process_image(image))
        out_clip.write_videofile(self.output_file, audio=False)

    def scale_image(self, image, scale_factor=3):
        return cv2.resize(image,
                          dsize=(self.image_size[1] // scale_factor, self.image_size[0] // scale_factor),
                          interpolation=cv2.INTER_CUBIC)

    def create_birds_eye_view(self, warped_image):
        """
        Create a image of the birds-eye view projection
        :param warped_image: warped binary image
        :return: RGB scaled image
        """

        # Stack the binary image and expand the color range
        warped_image = np.dstack((warped_image, warped_image, warped_image)) * 255
        overlay_large = self.lane.overlay(warped_image, draw_lines=True, fill_lane=False)

        # Resize
        return self.scale_image(overlay_large)

    @staticmethod
    def add_image_overlay(image, overlay, offset=(0, 0)):
        y_offset = offset[0]
        x_offset = offset[1]
        image[y_offset:y_offset + overlay.shape[0], x_offset:x_offset + overlay.shape[1]] = overlay
        return image

    def _process_image(self, org_image):
        # Create the warped binary image (birds-eye view with lane lines highlighted)
        undistorted_image = self.calibration.undistort(np.copy(org_image))
        warped_image = self.transform.transform(undistorted_image)
        warped_image, _ = threshold(warped_image, stack=False)

        # Update our lane tracker
        self.lane.update(warped_image=warped_image)

        # Add the lane overlay to our original image
        org_image = self.lane.overlay(org_image, transform=self.transform.invert())
        self.lane.overlay_curvature(org_image)

        # Create picture-in-picture overlays
        birds_eye_overlay = self.create_birds_eye_view(warped_image)
        binary_overlay = self.scale_image(np.dstack((warped_image, warped_image, warped_image)) * 255)

        # Overlay picture-in-picture style on original
        self.add_image_overlay(org_image, binary_overlay)
        self.add_image_overlay(org_image, birds_eye_overlay,
                               offset=(0, org_image.shape[1] - birds_eye_overlay.shape[1]))

        return org_image


def main():
    input_file = "test_videos/project_video.mp4"
    output_file = 'output_videos/processed_project_video.mp4'
    processor = VideoProcessor(input_file=input_file, output_file=output_file, image_size=(720, 1280))
    print("Processing video", input_file, output_file)
    # processor.process(sub_clip=(38, 42))
    # processor.process()
    processor.process(sub_clip=(0, 20))


if __name__ == '__main__':
    main()
