from AITrainingSharedLibrary.blend_img_with_mask import blend_image_and_mask
from typing import Optional, Dict, Union
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import polars as pl
import numpy as np
import logging




def save_results(
    video_segments: Dict[int, Dict[int, np.ndarray]],
    output_dir: Union[str, Path],
    logger: Optional[logging.Logger] = None,
    frame_df: Optional[pl.DataFrame] = None
) -> None:
    """
    Save the segmentation results for video frames.

    This function saves the original frames, segmentation masks, and blended images
    (original frame with mask overlay) to separate subdirectories within the specified
    output directory.

    Args:
        video_segments (Dict[int, Dict[int, np.ndarray]]): A dictionary containing
            frame indices as keys and another dictionary as values. The inner dictionary
            has object IDs as keys and corresponding segmentation masks as numpy arrays.
        output_dir (Union[str, Path]): The directory where results will be saved.
        logger (Optional[logging.Logger]): A logger object for logging information.
            If None, no logging will be performed.
        frame_df (Optional[pl.DataFrame]): A Polars DataFrame containing information
            about the video frames, including their file paths.

    Returns:
        None

    The function creates three subdirectories within the output directory:
    - 'orig': Contains the original video frames
    - 'masks': Contains the segmentation masks for each object in each frame
    - 'blended': Contains the original frames with the segmentation masks overlaid

    Each saved image is named according to its frame number and, for masks and
    blended images, the object ID.
    """
    # Create output directories
    output_dir = Path(output_dir)
    for subdir in ["orig", "masks", "blended"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Iterate over the frames and save the results
    num_frames = len(video_segments)
    save_res_prog_bar = tqdm(zip(frame_df.iter_rows(named=True), video_segments.items()), desc="Saving results", total=num_frames, leave=False)
    for row, (frame_idx, obj_masks) in save_res_prog_bar:
        # Get the original image
        save_res_prog_bar.set_postfix(f"Saving frame {frame_idx} of {num_frames}")
        orig_img = Image.open(row.get("img_path"))
        orig_img.save(output_dir / "orig" / f"frame_{frame_idx}.png")
        
        # Save the masks and blended images
        for obj_id, mask in obj_masks.items():
            mask = mask.squeeze()
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            blended_image = blend_image_and_mask(img_array=orig_img, mask_array=mask)
            mask_image.save(output_dir / "masks" / f"frame_{frame_idx}_object_{obj_id}.png")
            Image.fromarray(blended_image).save(output_dir / "blended" / f"frame_{frame_idx}_object_{obj_id}.png")
    
    if logger:
        logger.info(f"Results saved to {output_dir}")
    
    return None
