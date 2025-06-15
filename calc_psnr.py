import cv2
import numpy as np
import argparse


def compute_frame_psnr(frame1, frame2):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two frames.

    Args:
        frame1 (np.ndarray): First frame (RGB or grayscale)
        frame2 (np.ndarray): Second frame (must match frame1's shape and type)

    Returns:
        float: PSNR value in decibels, or None if invalid
    """
    # Ensure frames have the same shape
    if frame1.shape != frame2.shape:
        raise ValueError(f"Frame shapes do not match: {frame1.shape} vs {frame2.shape}")

    # Convert to float32 for PSNR calculation
    frame1 = frame1.astype(np.float32)
    frame2 = frame2.astype(np.float32)

    # Compute Mean Squared Error (MSE)
    mse = np.mean((frame1 - frame2) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match, no difference

    # Compute PSNR: 10 * log10(MAX^2 / MSE), where MAX is the max pixel value (255 for 8-bit)
    max_pixel = 255.0
    psnr = 10 * np.log10(max_pixel ** 2 / mse)
    return psnr


def compute_video_psnr(video_path1, video_path2):
    """
    Compute the average PSNR between two videos.

    Args:
        video_path1 (str): Path to the first video file
        video_path2 (str): Path to the second video file

    Returns:
        float: Average PSNR across all frames, or None if videos are incompatible
    """
    # Open the video files
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # Check if videos opened successfully
    if not cap1.isOpened():
        raise ValueError(f"Could not open video: {video_path1}")
    if not cap2.isOpened():
        raise ValueError(f"Could not open video: {video_path2}")

    # Get video properties
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count1 != frame_count2:
        cap1.release()
        cap2.release()
        raise ValueError(f"Frame counts do not match: {frame_count1} vs {frame_count2}")

    psnr_values = []
    frame_idx = 0

    # Process frames
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Break if either video ends
        if not ret1 or not ret2:
            break

        # Convert frames to RGB (OpenCV uses BGR by default)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # Compute PSNR for the current frame
        try:
            psnr = compute_frame_psnr(frame1, frame2)
            psnr_values.append(psnr)
        except ValueError as e:
            print(f"Error at frame {frame_idx}: {e}")
            cap1.release()
            cap2.release()
            return None

        frame_idx += 1

    # Release video objects
    cap1.release()
    cap2.release()

    # Check if any PSNR values were computed
    if not psnr_values:
        print("No valid frames to compare")
        return None

    # Compute average PSNR
    avg_psnr = np.mean(psnr_values)
    return avg_psnr


# Set up argument parser
parser = argparse.ArgumentParser(description="Compute the PSNR between two video files.")
parser.add_argument('--video_path1', type=str, required=True, help="Path to the first video file")
parser.add_argument('--video_path2', type=str, required=True, help="Path to the second video file")

# Parse arguments
args = parser.parse_args()

# Compute PSNR
try:
    avg_psnr = compute_video_psnr(args.video_path1, args.video_path2)
    if avg_psnr is not None:
        print(f"Average PSNR between {args.video_path1} and {args.video_path2}: {avg_psnr:.2f} dB")
    else:
        print("Failed to compute PSNR due to invalid frames or video mismatch")
except Exception as e:
    print(f"Error: {e}")
