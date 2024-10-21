from typing import Any, Tuple
import numpy as np
import torch.nn as nn 




def add_points_to_frame(
    sam2: nn.Module,
    inference_state: Any,
    points: np.ndarray,
    ann_frame_idx: int = 0,
    ann_obj_id: int = 1,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Add points to a specific frame for object segmentation using SAM2.

    Args:
        sam2 (nn.Module): The SAM2 model instance.
        inference_state (Any): The current inference state.
        points (np.ndarray): Array of points to add, shape (N, 2) where N is the number of points.
        ann_frame_idx (int, optional): The frame index to interact with. Defaults to 0.
        ann_obj_id (int, optional): Unique ID for the object being interacted with. Defaults to 1.

    Returns:
        Tuple: A tuple containing (_, out_obj_ids, out_mask_logits).
    """
    # For labels, `1` means positive click and `0` means negative click
    labels = np.ones(len(points), dtype=np.int32)
    
    _, out_obj_ids, out_mask_logits = sam2.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    
    return inference_state, out_obj_ids, out_mask_logits



def add_bbox_to_frame(
    sam2: nn.Module,
    inference_state: Any,
    bbox: np.ndarray,
    ann_frame_idx: int = 0,
    ann_obj_id: int = 1,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Add a bounding box to a specific frame for object segmentation using SAM2.

    Args:
        sam2 (nn.Module): The SAM2 model instance.
        inference_state (Any): The current inference state.
        bbox (np.ndarray): Array representing the bounding box, shape (4,) in format [x_min, y_min, x_max, y_max].
        ann_frame_idx (int, optional): The frame index to interact with. Defaults to 0.
        ann_obj_id (int, optional): Unique ID for the object being interacted with. Defaults to 1.

    Returns:
        Tuple: A tuple containing (inference_state, out_obj_ids, out_mask_logits).
    """
    _, out_obj_ids, out_mask_logits = sam2.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=bbox,
    )
    
    return inference_state, out_obj_ids, out_mask_logits