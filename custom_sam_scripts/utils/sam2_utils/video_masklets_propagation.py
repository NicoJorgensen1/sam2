import numpy as np
from typing import Dict, Optional, Any
import logging
import torch.nn as nn


def video_propagation(
    sam2: nn.Module,
    inference_state: Any,
    logger: Optional[logging.Logger] = None,
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Propagate masks across video frames using SAM2.

    This function takes the initial masks generated for the first frame and propagates
    them across subsequent frames in the video using the SAM2 model.

    Args:
        sam2 (nn.Module): The SAM2 model instance used for mask propagation.
        inference_state (Any): The current inference state containing initial masks and other relevant information.
        logger (logging.Logger): Logger instance for logging information about the propagation process.

    Returns:
        Dict[int, Dict[int, np.ndarray]]: A dictionary of video segments, where:
            - The outer keys are frame indices.
            - The inner keys are object IDs.
            - The values are numpy arrays representing binary masks for each object in each frame.

    Note:
        This function assumes that the `inference_state` object has a `mask_logits` attribute.
        If no masks are present in the inference state, an empty dictionary will be returned.
    """
    video_segments = {}
    if inference_state.mask_logits:
        if logger is not None:
            logger.info("Propagating masks across video")
        video_segments = {
            out_frame_idx: {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            for out_frame_idx, out_obj_ids, out_mask_logits in sam2.propagate_in_video(inference_state)
        }
    if logger is not None:
        logger.info(f"Processed {len(video_segments)} frames")
    return video_segments
