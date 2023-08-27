import cv2
import os
from tqdm import tqdm  
# Define constants
VIDEO_DIRECTORY = "//Users/alan/DEV/ibc_diffusion/dataset/raw/Suturing"  
OUTPUT_DIR = "/Users/alan/DEV/ibc_diffusion/dataset/preprocessed"
FRAME_SIZE = (224, 224)  # Update to the desired size for ViT

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loop through each video file
video_files = os.listdir(VIDEO_DIRECTORY)
for video_file in tqdm(video_files, desc="Processing Videos"):
    if video_file.endswith(".avi") and "Suturing" in video_file:
        video_path = os.path.join(VIDEO_DIRECTORY, video_file)
        output_subdir = os.path.join(OUTPUT_DIR, video_file[:-4])  
        os.makedirs(output_subdir, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Loop through the frames and save as resized JPEG images
        for frame_idx in tqdm(range(frame_count), desc=f"Processing {video_file}"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize the frame to the desired dimensions for ViT
            resized_frame = cv2.resize(frame, FRAME_SIZE)
            
            # Save the resized frame as a JPEG image
            frame_filename = os.path.join(output_subdir, f"frame{frame_idx:04d}.jpg")
            cv2.imwrite(frame_filename, resized_frame)

        # Release the video capture
        cap.release()

# Check the dimensions of preprocessed data
sample_frame = cv2.imread(os.path.join(output_subdir, video_files[0][:-4], "frame0000.jpg"))
print("Sample frame dimensions:", sample_frame.shape)
