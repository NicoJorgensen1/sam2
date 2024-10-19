from CameraAISharedLibrary.print_args_func import print_args 
from CameraAISharedLibrary.ensure_list_input import ensure_list
from CameraAISharedLibrary.extract_numbers_from_string import extract_numbers_from_string
from AITrainingSharedLibrary.mil_create_pl_dataframe import create_pl_df
from AITrainingSharedLibrary.get_relevant_dirs import add_dirs_to_path
from matplotlib import pyplot as plt
from typing import List, Optional
from tqdm import tqdm
import numpy as np
import polars as pl
import argparse
import cv2
import os

relevant_dirs = add_dirs_to_path()




def low_pass_on_CLAHE_frames(
        img_dir: List[str] = relevant_dirs.get("InfuSight_ENCXYN"),
        save_dir: Optional[str] = None,
        max_images: Optional[int] = None,
        **kwargs,
) -> None:
    img_df = ensure_list(create_pl_df(ds_paths=img_dir, images_only=True))[0]
    img_df = img_df.sort("frameNum", descending=False)
    if max_images is not None:
        rows_to_use = np.linspace(0, img_df.height - 1, max_images, dtype=int, endpoint=True)
        img_df = img_df.filter(pl.col("frameNum").is_in(img_df["frameNum"].to_numpy()[rows_to_use]))
    
    # Load the initial frame for subtraction and division operations
    initial_frame = cv2.imread(img_df["img_path"][0], cv2.IMREAD_GRAYSCALE)
    
    # Process each frame
    # Plot the results
    num_frames = len(img_df)
    fig, axes = plt.subplots(1, num_frames, figsize=(5*num_frames, 5))
    
    if num_frames == 1:
        axes = [axes]
    
    for i, row_dict in enumerate(tqdm(img_df.iter_rows(named=True), desc="Processing and plotting frames", total=num_frames)):
        # Read the current frame
        current_frame = cv2.imread(row_dict["img_path"], cv2.IMREAD_GRAYSCALE)
        
        # Apply low pass filter to the current frame
        low_pass_frame = cv2.GaussianBlur(current_frame, (5, 5), 0)
        
        # Subtract CLAHE of initial frame
        result = cv2.subtract(low_pass_frame, initial_frame)
        
        # Plot the frame
        axes[i].imshow(result, cmap='gray')
        axes[i].set_title(f"Frame {row_dict['frameNum']}")
        axes[i].axis('off')
    
    fig.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "low_pass_CLAHE_frames.png"))
    else:
        plt.show()
    
    return





### Run from CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images for preprocessing steps visualization.")
    parser.add_argument("--img_dir", type=str, default=relevant_dirs.get("InfuSight_ENCXYN"), help="Directory containing input images")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the output images")
    parser.add_argument("--max_images", type=str, default=None, help="Maximum number of images to process")
    args = parser.parse_args()

    # Edit the args 
    args.max_images = None if str(args.max_images).lower() == "none" else extract_numbers_from_string(inp_string=str(args.max_images), dtype=int, numbersWanted=1, return_all=False)

    # Print the input args 
    print_args(args=args, ljust_length=20, init_str="This is the input args when plotting preprocessing steps")

    # Run the function
    low_pass_on_CLAHE_frames(**vars(args))
