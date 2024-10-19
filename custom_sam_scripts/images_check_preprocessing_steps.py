from CameraAISharedLibrary.print_args_func import print_args 
from CameraAISharedLibrary.ensure_list_input import ensure_list
from CameraAISharedLibrary.extract_numbers_from_string import extract_numbers_from_string
from AITrainingSharedLibrary.mil_create_pl_dataframe import create_pl_df
from AITrainingSharedLibrary.get_relevant_dirs import add_dirs_to_path
from custom_sam_scripts.utils.preprocessing_methods import (
    subtract_initial_frame,
    divide_by_initial_frame,
    apply_low_high_pass,
    apply_clahe,
    scale_image,
    low_freq_enhancement,
    adaptive_threshold,
    difference_of_gaussians,
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




def plot_preprocessing_steps(
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

    # Titles for each subplot
    titles = ["Original Frame", "0th - Current Frame", "Current / 0th Frame", "CLAHE Frame"]

    functions_to_apply = [
        ("No func applied", lambda img: img),
        ("Low Pass", lambda img: apply_low_high_pass(img)[0]),
        ("Low Pass - init_frame", lambda img: apply_low_high_pass(img)[0] - initial_frame),
        ("Low Pass - low_pass(init_frame)", lambda img: apply_low_high_pass(img)[0] - apply_low_high_pass(initial_frame)[0]),
        ("High Pass", lambda img: apply_low_high_pass(img)[1]),
        ("High Pass - init_frame", lambda img: apply_low_high_pass(img)[1] - initial_frame),
        ("High Pass - high_pass(init_frame)", lambda img: apply_low_high_pass(img)[1] - apply_low_high_pass(initial_frame)[1]),
        ("Sobel", lambda img: cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)),
        ("Sobel - init_frame", lambda img: cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5) - initial_frame),
        ("Sobel - sobel(init_frame)", lambda img: cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5) - cv2.Sobel(initial_frame, cv2.CV_64F, 1, 1, ksize=5)),
        ("LoG", lambda img: cv2.Laplacian(cv2.GaussianBlur(img.astype(np.float64), (3, 3), 0), cv2.CV_64F)),
        ("LoG - init_frame", lambda img: cv2.Laplacian(cv2.GaussianBlur(img.astype(np.float64), (3, 3), 0), cv2.CV_64F) - initial_frame),
        ("LoG - LoG(init_frame)", lambda img: cv2.Laplacian(cv2.GaussianBlur(img.astype(np.float64), (3, 3), 0), cv2.CV_64F) -\
          cv2.Laplacian(cv2.GaussianBlur(initial_frame.astype(np.float64), (3, 3), 0), cv2.CV_64F)),        
        ("Fourier Transform", low_freq_enhancement),
        ("Fourier Transform - init_frame", lambda img: low_freq_enhancement(img) - initial_frame),
        ("Fourier Transform - Fourier(init_frame)", lambda img: low_freq_enhancement(img) - low_freq_enhancement(initial_frame)),
        ("Adaptive Threshold", adaptive_threshold),
        ("Adaptive Threshold - init_frame", lambda img: adaptive_threshold(img) - initial_frame),
        ("Adaptive Threshold - AdaptiveThreshold(init_frame)", lambda img: adaptive_threshold(img) - adaptive_threshold(initial_frame)),
        ("Difference of Gaussians", difference_of_gaussians),
        ("Difference of Gaussians - init_frame", lambda img: difference_of_gaussians(img) - initial_frame),
        ("Difference of Gaussians - DifferenceOfGaussians(init_frame)", lambda img: difference_of_gaussians(img) - difference_of_gaussians(initial_frame)),
    ]

    ### Plot each frame
    row_dicts = img_df.to_dicts()
    iterator = tqdm(enumerate(row_dicts), total=len(row_dicts), desc="Plotting preprocessing steps", leave=False)
    for img_path_idx, row_dict in iterator:
        iterator.set_description(f"Processing frame {row_dict.get('frameNum', img_path_idx+1)}")

        # Create a new figure for all frames with 5 rows and 4 columns
        num_rows, num_cols = len(functions_to_apply), len(titles)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(30, 30))

        # Load the frames
        current_frame = cv2.imread(row_dict["img_path"], cv2.IMREAD_GRAYSCALE)
        current_minus_initial = subtract_initial_frame(current_frame, initial_frame)
        current_divided_by_initial = divide_by_initial_frame(current_frame, initial_frame)
        current_clahe = apply_clahe(current_frame)

        frames = [initial_frame, current_frame, current_minus_initial, current_divided_by_initial, current_clahe]

        for row_idx, (func_name, func) in enumerate(functions_to_apply):
            for col_idx, (frame, title) in enumerate(zip(frames, titles)):
                # Process the frame 
                processed_frame = func(frame)
                processed_frame = scale_image(processed_frame)
                
                # Plot the frame
                axs[row_idx, col_idx].imshow(processed_frame, cmap='gray')
                axs[row_idx, col_idx].set_title(f"{func_name} on {title}")
                axs[row_idx, col_idx].axis('off')

                # Save the processed frame individually in the save_dir
                if save_dir is not None:
                    func_save_dir = os.path.join(save_dir, f"{func_name}")
                    os.makedirs(func_save_dir, exist_ok=True)
                    new_filepath = os.path.join(func_save_dir, f"{title}_{row_dict.get('frameNum', img_path_idx+1)}.png")
                    cv2.imwrite(new_filepath, processed_frame)

        fig.tight_layout()
        
        # Save the figure
        if save_dir is not None:
            fig.savefig(fname=os.path.join(save_dir, f"preprocessing_steps_frame_{row_dict.get('frameNum', img_path_idx+1)}.png"), bbox_inches='tight', dpi=300)
        
        # Close the figure to free up memory
        plt.close(fig)

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
    plot_preprocessing_steps(**vars(args))
