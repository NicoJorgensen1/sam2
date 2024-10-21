from CameraAISharedLibrary.print_args_func import print_args 
from CameraAISharedLibrary.ensure_list_input import ensure_list
from CameraAISharedLibrary.extract_numbers_from_string import extract_numbers_from_string
from CameraAISharedLibrary.apply_mold_masking import apply_masking
from AITrainingSharedLibrary.mil_create_pl_dataframe import create_pl_df
from AITrainingSharedLibrary.get_relevant_dirs import add_dirs_to_path
from custom_sam_scripts.utils.preprocessing_methods import (
    apply_clahe,
    divide_by_initial_frame,
    subtract_initial_frame,
    inverting_images,
    scale_image,
    gamma_transform,
    histogram_equalization,
)
from matplotlib import pyplot as plt
from typing import List, Optional
from tqdm import tqdm
import numpy as np
import polars as pl
import argparse
import cv2
import os

relevant_dirs = add_dirs_to_path()




def CLAHE_preprocessing(
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
    fig, axes = plt.subplots(2, num_frames, figsize=(5*num_frames, 10))
    
    if num_frames == 1:
        axes = axes.reshape(2, 1)
    
    for i, row_dict in enumerate(tqdm(img_df.iter_rows(named=True), desc="Processing and plotting frames", total=num_frames)):
        # Read the current frame in RGB and grayscale
        current_frame_rgb = cv2.cvtColor(cv2.imread(row_dict["img_path"]), cv2.COLOR_BGR2RGB)
        current_frame = cv2.imread(row_dict["img_path"], cv2.IMREAD_GRAYSCALE)
        mask = current_frame.copy()>0

        # Plot the original frame in RGB
        axes[0, i].imshow(current_frame_rgb)
        axes[0, i].set_title(f"Original Frame {row_dict['frameNum']}")
        axes[0, i].axis('off')

        # Subtract initial frame
        current_minus_init = cv2.subtract(current_frame, initial_frame)
        current_divided_by_init = divide_by_initial_frame(current_frame, initial_frame)
        
        # Apply low pass filter to the current frame
        low_pass_frame = cv2.GaussianBlur(current_minus_init, (5, 5), 0)

        clahe_frame = apply_clahe(current_divided_by_init)

        # Invert the image
        # clahe_frame = inverting_images(clahe_frame)

        # Apply mask
        clahe_frame = apply_masking(img_array=clahe_frame, mask_array=mask, mask_area="inner", frame_focus="PS")

        # Scale the image
        clahe_frame = scale_image(clahe_frame)

        # Apply histogram equalization
        clahe_frame = histogram_equalization(clahe_frame)

        # Apply gamma transform
        clahe_frame = gamma_transform(clahe_frame, gamma=0.5)
        
        # Apply mask
        clahe_frame = apply_masking(img_array=clahe_frame, mask_array=mask, mask_area="inner", frame_focus="PS")

        # clahe_frame = np.abs(clahe_frame).astype(np.uint8)

        # apply some otzu adaptive thresholding
        # clahe_frame = cv2.adaptiveThreshold(clahe_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # save the frame
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, f"CLAHE_frame_{row_dict['frameNum']}.png"), clahe_frame)

        
        # Plot the processed frame
        axes[1, i].imshow(clahe_frame, cmap='gray')
        axes[1, i].set_title(f"Processed Frame {row_dict['frameNum']}")
        axes[1, i].axis('off')
    
    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "CLAHE_frames_comparison.png"), bbox_inches='tight', dpi=300)
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
    CLAHE_preprocessing(**vars(args))
