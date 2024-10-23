from CameraAISharedLibrary.ensure_list_input import ensure_list
from CameraAISharedLibrary.apply_mold_masking import apply_masking
from CameraAISharedLibrary.crop2mask_func import crop2mask
from AITrainingSharedLibrary.mil_create_pl_dataframe import create_pl_df
from typing import Tuple, List, Optional
import polars as pl
import numpy as np
import cv2





def prepare_image_dataframe(
        img_dir: List[str],
        max_images: Optional[int] = None,
        accept_empty_init_frame: bool = True,
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    """
    Prepare the image dataframe for processing.

    Args:
        img_dir (List[str]): The directory containing the images.
        max_images (Optional[int]): The maximum number of images to process.

    Returns:
        Tuple[pl.DataFrame, Optional[pl.DataFrame]]: A tuple containing the main image dataframe and the initial frame dataframe (if it exists).
    """
    # Create the image dataframe
    img_df = ensure_list(create_pl_df(ds_paths=img_dir, images_only=True))[0]

    # Get the initial frame dataframe
    init_frame_df = img_df.filter(pl.col("frameNum") == 0)
    if init_frame_df.is_empty():
        if not accept_empty_init_frame:
            raise ValueError("No initial frame found in the image dataframe")
        init_frame_df = None
    
    # Filter out the initial frame and sort the remaining frames
    img_df = img_df.filter(pl.col("frameNum") != 0)
    img_df = img_df.sort("frameNum", descending=False)

    # Limit the number of images if max_images is provided
    if max_images is not None:
        rows_to_use = np.linspace(0, img_df.height - 1, max_images, dtype=int, endpoint=True)
        img_df = img_df.filter(pl.col("frameNum").is_in(img_df["frameNum"].to_numpy()[rows_to_use]))
    return img_df, init_frame_df



def get_initial_frame_and_mask(init_frame_df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and process the initial frame and its corresponding mask from a dataframe.

    This function reads the initial frame from the given dataframe, converts it to grayscale,
    applies masking, and crops the result to the mask area.

    Args:
        init_frame_df (pandas.DataFrame): A dataframe containing information about the initial frame,
                                          including the image path, frame focus, and mask filepath.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - init_frame_gray_masked (np.ndarray): The grayscale initial frame with masking applied and cropped to the mask area.
            - mask_array (np.ndarray): The binary mask array cropped to the same dimensions as the masked frame.

    Note:
        This function relies on several utility functions from the CameraAISharedLibrary,
        including apply_masking and crop2mask.
    """
    init_frame = cv2.cvtColor(cv2.imread(init_frame_df["img_path"][0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    init_frame_gray = cv2.cvtColor(init_frame, cv2.COLOR_RGB2GRAY)
    init_frame_focus = init_frame_df["frame_focus"].head(1).item()
    init_frame_mask_path = init_frame_df["seg_masks_filepath"][0]
    init_frame_gray_masked, mask_array = apply_masking(img_array=init_frame_gray, mask_path=init_frame_mask_path, mask_area="inner", frame_focus=init_frame_focus, return_mask=True)
    init_frame_gray_masked = crop2mask(img=init_frame_gray_masked, wanted_type=np.ndarray)
    mask_array = crop2mask(img=mask_array, wanted_type=np.ndarray)
    # resize the mask to the same size as the cropped initial frame
    mask_array = cv2.resize(mask_array.astype(np.uint8), (init_frame_gray_masked.shape[1], init_frame_gray_masked.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return init_frame_gray, init_frame_gray_masked, mask_array, init_frame_mask_path, init_frame_focus