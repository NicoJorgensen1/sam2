from CameraAISharedLibrary.print_args_func import print_args
from CameraAISharedLibrary.recursive_search import search_files
from AITrainingSharedLibrary.get_relevant_dirs import add_dirs_to_path
from AITrainingSharedLibrary.images_ensure_list_of_paths_input import get_list_of_single_filepaths
from AITrainingSharedLibrary.get_relevant_dirs import add_dirs_to_path
from typing import List, Union, Optional
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import torch
import os
import cv2

relevant_dirs = add_dirs_to_path()




def sam_video_inference(
        video_path_list: Union[str, Path, List[str], List[Path]] = relevant_dirs.get("InfuSight_ENJYJS_SHL05", os.getcwd()),
        model_size: str = "large",
        output_dir: Optional[str] = None,
        model_weights_dir: str = relevant_dirs.get("model_weights_dir", os.getenv("MODEL_WEIGHTS_DIR", "checkpoints")),
        config_dir: str = os.path.join(os.getcwd(), "sam2", "configs"),
        max_duration: int = 10,
) -> None:    
    # Get the list of single video filepaths
    video_path_list = get_list_of_single_filepaths(img_path_list=video_path_list)
    video_path_list = [video_path for video_path in video_path_list if video_path.endswith(("mp4", "mov", "avi", "mkv", "webm"))]
    video_path_list = filter_videos_by_duration(video_path_list, max_duration)
    if not video_path_list:
        raise ValueError(f"No videos found in {video_path_list} with max duration {max_duration}min")

    # Build the SAM2 video predictor
    model_paths = search_files(start_path=model_weights_dir, accepted_img_extensions=(".pt", ".pth"))
    sam_weights = [model_path for model_path in model_paths if "sam2" in model_path.lower() and model_size.lower() in model_path.lower()]
    if not sam_weights:
        raise ValueError(f"No SAM2 weights found for model size {model_size} in {model_weights_dir}")
    if len(sam_weights) > 1:
        raise ValueError(f"Found multiple SAM2 weights for model size {model_size} in {model_weights_dir}")
    checkpoint = sam_weights[0]
    
    # Find the model config checkpoint 
    model_cfg_list = [Path(config_path) for config_path in search_files(start_path=config_dir, accepted_img_extensions=(".yaml", ".yml"))]
    model_cfg_list = [model_cfg for model_cfg in model_cfg_list if "sam2.1" in str(model_cfg).lower() and model_size.lower()[0] in model_cfg.stem.lower()]
    if not model_cfg_list:
        raise ValueError(f"No SAM2 config found for model size {model_size} in {config_dir}")
    if len(model_cfg_list) > 1:
        raise ValueError(f"Found multiple SAM2 configs for model size {model_size} in {config_dir}")
    model_cfg = model_cfg_list[0]

    # Hydra config (which loads the model config) expects relative paths - for some reason without the relative path root
    relative_path = os.path.relpath(model_cfg, os.getcwd())
    root_removed = os.path.join(*relative_path.split(os.sep)[1:])

    # Build the SAM2 video predictor
    sam2 = build_sam2(config_file=str(root_removed), ckpt_path=checkpoint, device=get_device(), apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    # Run the SAM2 video predictor for each video path
    for video_path in video_path_list:
        # get the first frame of the video and let that be the image 
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Unable to read video file {video_path}")
        cap.release()

        masks = mask_generator.generate(frame)

    #     if output_dir is None:
    #         output_dir = video_path.parent

    #     with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

    #         state = predictor.init_state(str(video_path))

    #         # propagate the prompts to get masklets throughout the video
    #         for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
    #             pass 



def filter_videos_by_duration(video_path_list: List[Path], max_duration: int) -> List[Path]:
    """Filter videos based on max_duration."""
    filtered_video_path_list = []
    for video_path in tqdm(video_path_list, desc="Filtering videos by duration", leave=False):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            tqdm.write(f"Warning: Unable to open video file {video_path}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        
        if duration <= max_duration * 60:
            filtered_video_path_list.append(Path(video_path))
        else:
            tqdm.write(f"Skipping {video_path} as its duration ({duration:.2f}min) exceeds the maximum allowed duration ({max_duration}min)")
    
    return filtered_video_path_list



def get_device():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

    return device





### Run from CLI 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path_list", type=str, default=relevant_dirs.get("InfuSight_ENJYJS_SHL05", os.getcwd()), help="Path to the video file or directory containing video files")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory")
    parser.add_argument("--model_size", type=str, default="large", help="Model size to use for SAM2")
    parser.add_argument("--model_weights_dir", type=str, default=relevant_dirs.get("model_weights_dir", os.getenv("MODEL_WEIGHTS_DIR", "checkpoints")), help="Path to the model weights directory")
    parser.add_argument("--config_dir", type=str, default=os.path.join(os.getcwd(), "sam2", "configs"), help="Path to the model config directory")
    parser.add_argument("--max_duration", type=int, default=10, help="Maximum duration (in minutes) for each video")
    args = parser.parse_args()

    # Edit the input args 
    args.output_dir = None if "none" in str(args.output_dir).lower() else args.output_dir

    # Print the arguments
    print_args(args=args, ljust_length=20, init_str="This is the arguments when running SAM2 on a video")

    # Use vars to convert args to a dictionary and unpack it
    sam_video_inference(**vars(args))
