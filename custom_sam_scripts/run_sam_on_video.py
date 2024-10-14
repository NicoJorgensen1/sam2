from AITrainingSharedLibrary.get_relevant_dirs import add_dirs_to_path
from AITrainingSharedLibrary.images_ensure_list_of_paths_input import get_list_of_single_filepaths
from sam2.build_sam import build_sam2_video_predictor
from typing import List, Union
from pathlib import Path
import torch

relevant_dirs = add_dirs_to_path()


def sam_video_inference(
        video_path_list: Union[str, List[str]],
        model_size: str = "large",
) -> None:


    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    for video_path in video_path_list:

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path)

            # add new prompts and instantly get the output on the same frame
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, None)

            # propagate the prompts to get masklets throughout the video
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                ...