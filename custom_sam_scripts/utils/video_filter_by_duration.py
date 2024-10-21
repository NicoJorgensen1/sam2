from CameraAISharedLibrary.print_args_func import print_args
from CameraAISharedLibrary.ensure_list_input import ensure_list
from AITrainingSharedLibrary.mil_create_pl_dataframe import create_pl_df
from pathlib import Path
from typing import List
from tqdm import tqdm
import argparse
import cv2




def filter_videos_by_duration(
    video_path_list: List[Path],
    max_duration: int,
    verbose: bool = False,
) -> List[Path]:
    """
    Filter a list of video paths based on their duration.

    This function processes a list of video file paths, checks the duration of each video,
    and returns a new list containing only the videos with a duration less than or equal
    to the specified maximum duration.

    Args:
        video_path_list (List[Path]): A list of Path objects representing video file paths.
        max_duration (int): The maximum allowed duration for videos in minutes.
        verbose (bool, optional): If True, print messages about skipped videos. Defaults to False.

    Returns:
        List[Path]: A list of Path objects representing video files that meet the duration criteria.

    Note:
        - This function uses OpenCV to read video properties.
        - Videos that cannot be opened will be skipped with a warning message.
        - The duration is calculated as (frame count / frames per second).
        - The function uses tqdm to show a progress bar during processing.
    """
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
        elif verbose:
            tqdm.write(f"Skipping {video_path} as its duration ({duration:.2f}min) exceeds the maximum allowed duration ({max_duration}min)")

    return filtered_video_path_list





### Run from CLI 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter videos based on maximum duration.")
    parser.add_argument("--directory", type=str, help="Directory containing video files")
    parser.add_argument("--max_duration", type=int, default=10, help="Maximum duration in minutes (default: 10)")
    args = parser.parse_args()

    # Print the args
    print_args(args=args, init_str="This is the input args chosen when filtering videos based on duration")

    # Run the function
    file_df = ensure_list(create_pl_df(ds_paths=args.directory, videos_only=True))[0]
    filtered_videos = filter_videos_by_duration(video_path_list=file_df["img_path"].to_list(), max_duration=args.max_duration)

    # Print the filtered videos
    for video_path in filtered_videos:
        print(video_path)
