import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from camera_calibration import default_camera_calibration
from transform import PerspectiveTransform
from thresholding import threshold

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


def process_image(org_image):
    # Run the pipeline on a copy of the image
    image = calibration.undistort(np.copy(org_image))
    # image = transform.transform(image)
    combined, color = threshold(image, stack=True)
    combined = transform.transform(combined)

    # Stack the binary image and expand the color range
    color_combined = np.dstack((combined, combined, combined)) * 255

    # Resize
    color_combined_small = cv2.resize(color_combined, dsize=(width // 2, height // 2), interpolation=cv2.INTER_CUBIC)

    # Overlay result picture-in-picture style on original
    y_offset, x_offset = 0, 0
    org_image[y_offset:y_offset + color_combined_small.shape[0], x_offset:x_offset + color_combined_small.shape[1]] = color_combined_small

    return org_image


print("Processing video")
output = 'output_videos/processed_project_video.mp4'
clip = VideoFileClip("test_videos/project_video.mp4").subclip(38,42)
out_clip = clip.fl_image(process_image)
out_clip.write_videofile(output, audio=False)