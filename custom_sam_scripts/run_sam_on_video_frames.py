from CameraAISharedLibrary.print_args_func import print_args
from CameraAISharedLibrary.ensure_list_input import ensure_list
from CameraAISharedLibrary.extract_numbers_from_string import extract_numbers_from_string
from AITrainingSharedLibrary.get_relevant_dirs import add_dirs_to_path
from AITrainingSharedLibrary.mil_create_pl_dataframe import create_pl_df
from AITrainingSharedLibrary.setup_logger_func import setup_logger
from custom_sam_scripts.utils.get_sam_for_video_inference import get_sam_for_video_inference
from custom_sam_scripts.utils.save_sam2_masks import save_results
from custom_sam_scripts.add_points_or_bboxes_to_frame import add_points_and_bboxes
from typing import List, Optional, Union, Dict
from pathlib import Path
import numpy as np
import argparse
import os
relevant_dirs = add_dirs_to_path()




def run_sam_on_video_frames(
    frame_path_list: List[Path],
    model_size: str,
    model_weights_dir: str,
    config_dir: str,
    points: List[np.ndarray],
    neg_points: List[np.ndarray],
    bboxes: List[np.ndarray],
    save_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Run SAM2 (Segment Anything Model 2) on a list of video frames.

    This function processes a list of video frames using SAM2, initializes the model,
    adds points to the first frame, propagates masks across the video, and optionally
    saves the results.

    Args:
        frame_path_list (List[Path]): List of paths to video frames.
        model_size (str): Size of the SAM2 model (e.g., "large", "base").
        model_weights_dir (str): Directory containing SAM2 model weights.
        config_dir (str): Directory containing SAM2 model configuration files.
        points (List[np.ndarray]): List of arrays of points to add to the first frame, shape (N, 2).
        neg_points (List[np.ndarray]): List of arrays of negative points to add to the first frame, shape (N, 2).
        bboxes (List[np.ndarray]): List of arrays of bounding boxes to add to the first frame, shape (N, 4).
        save_dir (Optional[Union[str, Path]], optional): Directory to save results. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Dict[int, Dict[int, np.ndarray]]: Dictionary of video segments, where keys are
        frame indices and values are dictionaries of object IDs and their mask arrays.

    Example:
        >>> frame_paths = [Path("frame_001.jpg"), Path("frame_002.jpg"), ...]
        >>> points = np.array([[300, 400], [500, 600]])
        >>> neg_points = np.array([[100, 100], [200, 200]])
        >>> bboxes = np.array([[100, 200, 300, 400], [500, 600, 700, 800]])
        >>> segments = run_sam_on_video_frames(frame_paths, "large", "weights/", "configs/", points, neg_points, bboxes, "results/")

    Note:
        - The function uses a dataframe to process frames, initializes SAM2, and propagates masks.
        - It logs various steps and information using a logger.
        - Results are saved only if a save_dir is provided.
    """
    # Start logging
    # Create a logger to store logs in the current working directory (20 = INFO)
    logger = setup_logger(log_path=os.getcwd(), level=20, name="sam_video_processing.log", force_create_log_path=True)
    logger.info("Starting SAM2 video segmentation")
    
    # Get a dataframe of the frames
    frame_df = create_pl_df(ds_paths=frame_path_list, images_only=True)
    if frame_df is None:
        raise ValueError("The dataframe of the frames is None")
    frame_df = ensure_list(frame_df)[0].sort("frameNum", descending=False)
    logger.info(f"Processed {len(frame_df)} frames")

    # Get the SAM2 model
    sam2 = get_sam_for_video_inference(model_size, model_weights_dir, config_dir, auto_mask_generator=False)
    logger.info(f"Initialized SAM2 model with size {model_size}")

    # Set the inference state
    inference_state = sam2.init_state(video_path=frame_path_list)
    sam2.reset_state(inference_state)
    logger.info("Initialized inference state")

    # Add points and bboxes to the frame
    inference_state, out_mask_logits = add_points_and_bboxes(sam2, inference_state, points, bboxes, neg_points, frame_idx=0)

    # Propagate the masks across the video on the rest of the frames
    video_segments = {}
    if out_mask_logits:
        logger.info("Propagating masks across video")
        video_segments = {out_frame_idx: {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)} 
                                for out_frame_idx, out_obj_ids, out_mask_logits in sam2.propagate_in_video(inference_state)}
    logger.info(f"Processed {len(video_segments)} frames")


    # Save the results
    if save_dir is not None and video_segments:
        save_results(video_segments=video_segments, output_dir=save_dir, frame_df=frame_df)

    logger.info("SAM2 video segmentation completed")
    return video_segments





### Run from CLI 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path_list", type=str, default=os.path.join(relevant_dirs.get("InfuSight_ENJYJS_SHL05", os.getcwd()), "Masked_Frames"),
                        help="Path to the frame file or directory containing frame files")
    parser.add_argument("--model_size", type=str, default="large", help="Model size to use for SAM2")
    parser.add_argument("--model_weights_dir", type=str, default=relevant_dirs.get("model_weights_dir", os.getenv("MODEL_WEIGHTS_DIR", "checkpoints")), help="Path to the model weights directory")
    parser.add_argument("--config_dir", type=str, default=os.path.join(os.getcwd(), "sam2", "configs"), help="Path to the model config directory")
    parser.add_argument("--points", type=str, default="300 900", help="List of points in 'x,y' format")
    parser.add_argument("--neg_points", type=str, default=None, help="List of negative points in 'x,y' format")
    parser.add_argument("--bboxes", type=str, default=None, help="List of bboxes in 'x1,y1,x2,y2' format")
    parser.add_argument("--save_dir", type=str, default=None, help="Path to the save directory")
    args = parser.parse_args()

    # Edit the input args 
    args.save_dir = None if "none" in str(args.save_dir).lower() else args.save_dir

    # Process user-provided points and bboxes
    if "none" not in str(args.points).lower():
        args.points = args.points.split(",")
        args.points = np.asarray([extract_numbers_from_string(inp_string=p, return_all=True, dtype=int) for p in args.points])
    else:
        args.points = None
    if "none" not in str(args.neg_points).lower():
        args.neg_points = args.neg_points.split(",")
        args.neg_points = np.asarray([extract_numbers_from_string(inp_string=p, return_all=True, dtype=int) for p in args.neg_points])
    else:
        args.neg_points = None
    if "none" not in str(args.bboxes).lower():
        args.bboxes = args.bboxes.split(",")
        args.bboxes = np.asarray([extract_numbers_from_string(inp_string=b, return_all=True, dtype=int) for b in args.bboxes])
    else:
        args.bboxes = None

    # Print the arguments
    print_args(args=args, init_str="This is the arguments when running SAM2 on a video")

    # Run the script
    video_segments = run_sam_on_video_frames(**vars(args))



