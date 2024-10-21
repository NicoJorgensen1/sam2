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




def new_preprocessing(
        img_dir: List[str] = os.path.join(relevant_dirs.get("InfuSight_ENCXYN_SHL01"), "Frames"),
        save_dir: Optional[str] = os.path.join(relevant_dirs.get("InfuSight_ENCXYN_SHL01"), "New_preprocessing"),
        max_images: Optional[int] = None,
        **kwargs,
) -> None:
    # Get the image dataframe
    img_df = ensure_list(create_pl_df(ds_paths=img_dir, images_only=True))[0]
    init_frame_df = img_df.filter(pl.col("frameNum") == 0)
    img_df = img_df.filter(pl.col("frameNum") != 0)
    img_df = img_df.sort("frameNum", descending=False)
    if max_images is not None:
        rows_to_use = np.linspace(0, img_df.height - 1, max_images, dtype=int, endpoint=True)
        img_df = img_df.filter(pl.col("frameNum").is_in(img_df["frameNum"].to_numpy()[rows_to_use]))
    
    # Get the initial frame
    init_frame = cv2.cvtColor(cv2.imread(init_frame_df["img_path"][0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    init_frame_gray = cv2.cvtColor(init_frame, cv2.COLOR_RGB2GRAY)
    
    # Iterate through each frame 
    frame_prog_bar = tqdm(enumerate(img_df.iter_rows(named=True)), desc="Processing frames", total=img_df.height)
    for img_idx, row_dict in frame_prog_bar:
        frame_prog_bar.set_description(f"Processing frame {row_dict.get('frameNum', img_idx):04d} of {img_df.height}")

        # Read the current frame
        current_frame = cv2.cvtColor(cv2.imread(row_dict.get("img_path"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        
        # Divide the current frame by the initial frame
        current_frame_preprocessed = current_frame_gray / (init_frame_gray+1e-10)
        current_frame_preprocessed = cv2.normalize(current_frame_preprocessed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U).astype(np.uint8)
        
        # Save the current frame
        save_path = os.path.join(save_dir, f"{row_dict.get('frameNum', img_idx):04d}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, current_frame_preprocessed)

    return None





### Run from CLI ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default=os.path.join(relevant_dirs.get("InfuSight_ENCXYN_SHL01"), "Frames"), help="The directory containing the images to process.")
    parser.add_argument("--save_dir", type=str, default=None, help="The directory to save the output images to.")
    parser.add_argument("--max_images", type=int, default=None, help="The maximum number of images to process.")
    args = parser.parse_args()

    # Edit the args 
    args.max_images = None if str(args.max_images).lower() == "none" else extract_numbers_from_string(inp_string=str(args.max_images), dtype=int, numbersWanted=1, return_all=False)

    # Print the arguments
    print_args(args=args, init_str="Running new preprocessing")

    # Run the function
    new_preprocessing(img_dir=args.img_dir, save_dir=args.save_dir)
