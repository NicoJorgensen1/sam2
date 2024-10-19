from CameraAISharedLibrary.print_args_func import print_args
from CameraAISharedLibrary.ensure_list_input import ensure_list
from AITrainingSharedLibrary.get_relevant_dirs import add_dirs_to_path
from AITrainingSharedLibrary.mil_create_pl_dataframe import create_pl_df
from custom_sam_scripts.utils.get_sam_for_video_inference import get_sam_for_video_inference
from typing import List
from pathlib import Path
import argparse
import os
relevant_dirs = add_dirs_to_path()




def run_sam_on_video_frames(
    frame_path_list: List[Path],
    output_dir: Path,
    model_size: str,
    model_weights_dir: str,
    config_dir: str,
    **kwargs,
) -> None:
    # Get a dataframe of the frames
    frame_df = create_pl_df(ds_paths=frame_path_list, images_only=True)
    if frame_df is None:
        raise ValueError("The dataframe of the frames is None")
    frame_df = ensure_list(frame_df)[0].sort("frameNum", descending=False)

    # Get the SAM2 model
    sam2 = get_sam_for_video_inference(model_size, model_weights_dir, config_dir, auto_mask_generator=False)

    # Set the inference state
    inference_state = sam2.init_state(video_path=frame_path_list)




### Run from CLI 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path_list", type=str, default=relevant_dirs.get("InfuSight_ENJYJS_SHL05", os.getcwd()), help="Path to the frame file or directory containing frame files")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory")
    parser.add_argument("--model_size", type=str, default="large", help="Model size to use for SAM2")
    parser.add_argument("--model_weights_dir", type=str, default=relevant_dirs.get("model_weights_dir", os.getenv("MODEL_WEIGHTS_DIR", "checkpoints")), help="Path to the model weights directory")
    parser.add_argument("--config_dir", type=str, default=os.path.join(os.getcwd(), "sam2", "configs"), help="Path to the model config directory")
    args = parser.parse_args()

    # Edit the input args 
    args.output_dir = None if "none" in str(args.output_dir).lower() else args.output_dir

    # Print the arguments
    print_args(args=args, ljust_length=20, init_str="This is the arguments when running SAM2 on a video")

    # Run the script
    run_sam_on_video_frames(**vars(args))