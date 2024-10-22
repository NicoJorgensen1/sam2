from CameraAISharedLibrary.print_args_func import print_args 
from CameraAISharedLibrary.ensure_list_input import ensure_list
from CameraAISharedLibrary.str2bool_func import str2bool
from CameraAISharedLibrary.extract_numbers_from_string import extract_numbers_from_string
from CameraAISharedLibrary.apply_mold_masking import apply_masking
from CameraAISharedLibrary.crop2mask_func import crop2mask
from AITrainingSharedLibrary.mil_create_pl_dataframe import create_pl_df
from AITrainingSharedLibrary.get_relevant_dirs import add_dirs_to_path
from AITrainingSharedLibrary.images_build_video_from_images import build_video_from_images
from custom_sam_scripts.utils.preprocessing_methods import (
    divide_by_initial_frame,
    scale_image,
    gamma_transform,
)
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
        do_preprocessing: bool = True,
        build_video: bool = True,
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
    
    # Get the initial frame and mold line mask
    init_frame = cv2.cvtColor(cv2.imread(init_frame_df["img_path"][0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    init_frame_gray = cv2.cvtColor(init_frame, cv2.COLOR_RGB2GRAY)
    init_frame_focus = init_frame_df["frame_focus"].head(1).item()
    init_frame_mask_path = init_frame_df["seg_masks_filepath"][0]
    init_frame_gray_masked, mask_array = apply_masking(img_array=init_frame_gray, mask_path=init_frame_mask_path, mask_area="inner", frame_focus=init_frame_focus, return_mask=True)
    init_frame_gray_masked = crop2mask(img=init_frame_gray_masked, wanted_type=np.ndarray)
    mask_array = crop2mask(img=mask_array, wanted_type=np.ndarray)
    
    # Iterate through each frame 
    os.makedirs(save_dir, exist_ok=True)
    images_saved = []
    frame_prog_bar = tqdm(enumerate(img_df.iter_rows(named=True)), desc="Processing frames", total=img_df.height)
    for img_idx, row_dict in frame_prog_bar:
        frame_num = row_dict.get('frameNum', img_idx)
        frame_prog_bar.set_description(f"Processing frameNum {frame_num:04d} which is image {img_idx+1:d} of {img_df.height}")

        # Read the current frame
        current_frame = cv2.cvtColor(cv2.imread(row_dict.get("img_path"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        current_frame_gray_masked = apply_masking(img_array=current_frame_gray, mask_path=init_frame_mask_path, mask_area="inner", frame_focus=init_frame_focus)
        current_frame_gray_masked = crop2mask(img=current_frame_gray_masked, wanted_type=np.ndarray)
        
        # Divide the current frame by the initial frame and scale the result to 0-255
        if do_preprocessing:
            current_frame_preprocessed = divide_by_initial_frame(current_frame_gray_masked, init_frame_gray_masked, mask=mask_array)
            current_frame_preprocessed = scale_image(current_frame_preprocessed, mask=mask_array)
            current_frame_preprocessed = gamma_transform(image=current_frame_preprocessed, gamma=0.50, mask=mask_array).astype(np.uint8)
        else:
            current_frame_preprocessed = current_frame_gray_masked
        
        # Save the preprocessed frame
        save_path = os.path.join(save_dir, f"processed_{row_dict.get('basename')}")
        cv2.imwrite(save_path, current_frame_preprocessed)
        images_saved.append(save_path)
    
    # Build a video from the saved images
    if build_video:
        video_path = build_video_from_images(img_paths=images_saved, save_path=os.path.join(save_dir, "processed_video.mp4"), fps=25)
        print(f"Video saved to {video_path}")
    return images_saved





### Run from CLI ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default=os.path.join(relevant_dirs.get("InfuSight_ENCXYN_SHL01"), "Frames"), help="The directory containing the images to process.")
    parser.add_argument("--save_dir", type=str, default=None, help="The directory to save the output images to.")
    parser.add_argument("--max_images", type=str, default=None, help="The maximum number of images to process.")
    parser.add_argument("--do_preprocessing", type=str2bool, default=True, help="Whether to perform preprocessing on the images.")
    parser.add_argument("--build_video", type=str2bool, default=True, help="Whether to build a video from the processed images.")
    args = parser.parse_args()

    # Edit the args 
    args.max_images = None if str(args.max_images).lower() == "none" else extract_numbers_from_string(inp_string=str(args.max_images), dtype=int, numbersWanted=1, return_all=False)

    # Print the arguments
    print_args(args=args, init_str="Running new preprocessing")

    # Run the function
    new_preprocessing(img_dir=args.img_dir, save_dir=args.save_dir, max_images=args.max_images)
