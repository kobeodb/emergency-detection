import cv2
import os


def video_to_frames(video_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break  # Exit the loop when no frames are left

        # Save the frame as an image file
        frame_filename = os.path.join(output_dir, f"frame_{count}.jpg")
        cv2.imwrite(frame_filename, frame)

        count += 1

    video.release()
    print(f"Saved {count} frames to {output_dir}")


# Example usage:
video_to_frames('stroke_simulation_1.mp4', '../data/ground_truth')
