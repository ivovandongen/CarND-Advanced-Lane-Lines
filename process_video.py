import numpy as np
from moviepy.editor import VideoFileClip
from camera_calibration import load_default_camera_calibration
from thresholding import threshold


# Prepare camera calibration
print("Calibrating camera")
calibration = calibration()


def process_image(image):
    image = calibration.undistort(image)
    combined, color = threshold(image, stack=True)
    # return color
    return np.dstack((combined, combined, combined)) * 255


print("Processing video")
output = 'output_videos/processed_challenge_video.mp4'
clip = VideoFileClip("test_videos/challenge_video.mp4").subclip(0,5)
out_clip = clip.fl_image(process_image)
out_clip.write_videofile(output, audio=False)