import cv2
import os

# Specify the path to your video file
video_path = "C:/Users/HP/Downloads/My_senior/data_lane/video_lane.mp4"

# Specify the folder where you want to save the extracted images
output_folder = "C:/Users/HP/Downloads/My_senior/data_lane/frames_num"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Total frames: {frame_count}, FPS: {fps}")

# Loop through each frame and save it as an image
for frame_number in range(frame_count):
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame.")
        break

    # Save frame as an image
    image_path = os.path.join(output_folder, f"{frame_number:04d}.png")
    cv2.imwrite(image_path, frame)

    # Display progress
    if frame_number % 100 == 0:
        print(f"Processed {frame_number} frames")

# Release the video capture object
cap.release()
