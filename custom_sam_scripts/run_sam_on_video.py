from CameraAISharedLibrary.print_args_func import print_args
from AITrainingSharedLibrary.get_relevant_dirs import add_dirs_to_path
from AITrainingSharedLibrary.images_ensure_list_of_paths_input import get_list_of_single_filepaths
from AITrainingSharedLibrary.get_relevant_dirs import add_dirs_to_path
from sam2.build_sam import build_sam2_video_predictor
from typing import List, Union, Optional
from pathlib import Path
import argparse
import torch
import os

relevant_dirs = add_dirs_to_path()




def sam_video_inference(
        video_path_list: Union[str, Path, List[str], List[Path]] = relevant_dirs.get("InfuSight_ENJYJS_SHL05", os.getcwd()),
        model_size: str = "large",
        output_dir: Optional[str] = None,
) -> None:
    
    # Get the list of single video filepaths
    video_path_list = get_list_of_single_filepaths(video_path_list=video_path_list)
    video_path_list = [Path(video_path) for video_path in video_path_list]

    # Build the SAM2 video predictor
    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    # Run the SAM2 video predictor for each video path
    for video_path in video_path_list:
        if output_dir is None:
            output_dir = video_path.parent
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            
            state = predictor.init_state(video_path)

            # add new prompts and instantly get the output on the same frame
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, None)

            # propagate the prompts to get masklets throughout the video
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                pass 





### Run from CLI 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=relevant_dirs.get("InfuSight_ENJYJS_SHL05", os.getcwd()), help="Path to the video file or directory containing video files")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory")
    args = parser.parse_args()

    # Edit the input args 
    args.output_dir = None if "none" in str(args.output_dir).lower() else args.output_dir

    # Print the arguments
    print_args(args=args, ljust_length=20, init_str="This is the arguments when running SAM2 on a video")

    sam_video_inference(args.video_path, args.output_dir)

