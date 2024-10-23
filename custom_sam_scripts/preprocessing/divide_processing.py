from CameraAISharedLibrary.print_args_func import print_args 
from CameraAISharedLibrary.str2bool_func import str2bool
from CameraAISharedLibrary.extract_numbers_from_string import extract_numbers_from_string
from CameraAISharedLibrary.apply_mold_masking import apply_masking
from CameraAISharedLibrary.crop2mask_func import crop2mask
from AITrainingSharedLibrary.get_relevant_dirs import add_dirs_to_path
from AITrainingSharedLibrary.images_build_video_from_images import build_video_from_images
from AITrainingSharedLibrary.blend_img_with_mask import overlay_mask_on_image
from custom_sam_scripts.utils.get_init_frame_and_mask_from_df import prepare_image_dataframe, get_initial_frame_and_mask
from custom_sam_scripts.utils.plot_save_images import plot_and_save_images
from custom_sam_scripts.utils.get_largest_blob import select_largest_component
from custom_sam_scripts.utils.preprocessing_methods import (
    divide_by_initial_frame,
    scale_image,
    gamma_transform,
    invert_image,
)
from typing import List, Optional
from scipy import ndimage
from tqdm import tqdm
import numpy as np
import argparse
import cv2
import os

relevant_dirs = add_dirs_to_path()




def new_preprocessing(
        img_dir: List[str] = os.path.join(relevant_dirs.get("InfuSight_ENCXYN_SHL01"), "Frames"),
        save_dir: Optional[str] = os.path.join(relevant_dirs.get("InfuSight_ENCXYN_SHL01"), "New_preprocessing"),
        max_images: Optional[int] = None,
        do_preprocessing: bool = True,
        threshold_preprocessing: bool = False,
        plot_blended_frame: bool = True,
        build_video: bool = True,
        **kwargs,
) -> None:
    # Get the initial stuff
    img_df, init_frame_df = prepare_image_dataframe(img_dir, max_images, accept_empty_init_frame=False)
    init_frame_gray, init_frame_gray_masked, mask_array, init_frame_mask_path, init_frame_focus = get_initial_frame_and_mask(init_frame_df)
    
    # Iterate through each frame 
    os.makedirs(save_dir, exist_ok=True)
    processed_images_saved = []
    frame_prog_bar = tqdm(enumerate(img_df.iter_rows(named=True)), desc="Processing frames", total=img_df.height)
    for img_idx, row_dict in frame_prog_bar:
        frame_num = row_dict.get('frameNum', img_idx)
        frame_prog_bar.set_description(f"Processing frameNum {frame_num:04d} which is image {img_idx+1:d} of {img_df.height}")

        # Read the current frame
        current_frame_color = cv2.cvtColor(cv2.imread(row_dict.get("img_path"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        current_frame_color_masked = apply_masking(img_array=current_frame_color, mask_path=init_frame_mask_path, mask_area="inner", frame_focus=init_frame_focus)
        current_frame_color_masked = crop2mask(img=current_frame_color_masked, wanted_type=np.ndarray)
        current_frame_gray_masked = cv2.cvtColor(current_frame_color_masked, cv2.COLOR_RGB2GRAY)

        # Divide the current frame by the initial frame and scale the result to 0-255
        if do_preprocessing:
            current_frame_preprocessed = divide_by_initial_frame(current_frame_gray_masked, init_frame_gray_masked, mask=None)
            if threshold_preprocessing:
                # Do the preprocessing steps
                current_frame_preprocessed = np.where(current_frame_preprocessed >= 1, 1, current_frame_preprocessed)
                min_val = np.min(current_frame_preprocessed[mask_array>0])
                max_val = np.max(current_frame_preprocessed[mask_array>0])
                current_frame_preprocessed = ((current_frame_preprocessed - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                current_frame_preprocessed = invert_image(img=current_frame_preprocessed, mask=mask_array)
                current_frame_preprocessed = apply_masking(img_array=current_frame_preprocessed, mask_array=mask_array, mask_area="inner", frame_focus=init_frame_focus)
                current_frame_preprocessed = np.clip(current_frame_preprocessed, a_min=0, a_max=255).astype(np.uint8)
                
                # Create a mask of the processed frame
                if plot_blended_frame:
                    current_frame_mask = np.where(current_frame_preprocessed > 20, 1, 0).astype(bool)

                    # use morphological operations to clean up the mask
                    current_frame_mask = cv2.morphologyEx(current_frame_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
                    current_frame_mask = cv2.morphologyEx(current_frame_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
                    current_frame_mask = ndimage.binary_fill_holes(current_frame_mask).astype(int)
                    current_frame_mask = select_largest_component(current_frame_mask)
                    current_frame_mask = ndimage.binary_fill_holes(current_frame_mask).astype(int)
                    blended_processed_frame = overlay_mask_on_image(img_array=current_frame_color_masked, mask_array=current_frame_mask, id2color={1: (255, 0, 0)}, alpha=0.5)

                    # Plot and save the blended frame
                    plot_and_save_images(images=[current_frame_color_masked, current_frame_preprocessed, blended_processed_frame], 
                                        titles=["Original frame", "Processed frame", "Blended processed frame"],
                                        save_path=os.path.join(save_dir, f"processed_and_blended_{row_dict.get('basename')}"))
            else:
                current_frame_preprocessed = scale_image(current_frame_preprocessed, mask=mask_array)
                current_frame_preprocessed = gamma_transform(image=current_frame_preprocessed, gamma=0.50, mask=mask_array).astype(np.uint8)
        else:
            current_frame_preprocessed = crop2mask(img=apply_masking(img_array=current_frame_color, 
                                                                     mask_path=init_frame_mask_path, 
                                                                     mask_area="inner", 
                                                                     frame_focus=init_frame_focus), 
                                                                     wanted_type=np.ndarray)
        
        # Save the preprocessed frame
        save_path = os.path.join(save_dir, f"processed_{row_dict.get('basename')}")
        cv2.imwrite(save_path, current_frame_preprocessed.astype(np.uint8))
        processed_images_saved.append(save_path)
    
    # Build a video from the saved images
    if build_video:
        video_path = build_video_from_images(img_paths=processed_images_saved, save_path=os.path.join(save_dir, "processed_video.mp4"), fps=25)
        print(f"Video saved to {video_path}")
    return processed_images_saved





### Run from CLI ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default=os.path.join(relevant_dirs.get("InfuSight_ENCXYN_SHL01"), "Frames"), help="The directory containing the images to process.")
    parser.add_argument("--save_dir", type=str, default=None, help="The directory to save the output images to.")
    parser.add_argument("--max_images", type=str, default=None, help="The maximum number of images to process.")
    parser.add_argument("--do_preprocessing", type=str2bool, default=True, help="Whether to perform preprocessing on the images.")
    parser.add_argument("--threshold_preprocessing", type=str2bool, default=False, help="Whether to perform the threshold processing")
    parser.add_argument("--plot_blended_frame", type=str2bool, default=True, help="Whether to plot and save a blended frame.")
    parser.add_argument("--build_video", type=str2bool, default=True, help="Whether to build a video from the processed images.")
    args = parser.parse_args()

    # Edit the args 
    args.max_images = None if str(args.max_images).lower() == "none" else extract_numbers_from_string(inp_string=str(args.max_images), dtype=int, numbersWanted=1, return_all=False)

    # Print the arguments
    print_args(args=args, init_str="Running new preprocessing")

    # Run the function
    new_preprocessing(**vars(args))
