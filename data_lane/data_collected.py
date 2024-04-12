import cv2
import os
from datetime import datetime

# Open the video file
video_path = 'video_lane.mp4'
cap = cv2.VideoCapture(video_path)

# Create a directory to store the extracted frames
output_directory = 'frames'
os.makedirs(output_directory, exist_ok=True)

# Loop through each frame and save it as an image
frame_count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Save the frame as an image
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '')
    frame_filename = os.path.join(output_directory, f'frame_{timestamp}.jpg')
    cv2.imwrite(frame_filename, frame)

# Release the video capture object
cap.release()

print(f'Frames extracted successfully. Total frames: {frame_count}')