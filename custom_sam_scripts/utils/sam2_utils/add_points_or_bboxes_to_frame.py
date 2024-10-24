from typing import Any, Tuple
import numpy as np
import torch.nn as nn 




def add_points_and_bboxes(
    sam2: nn.Module,
    inference_state: Any,
    points: np.ndarray,
    bboxes: np.ndarray,
    neg_points: np.ndarray,
    frame_idx: int = 0,
    obj_id: int = 1,
) -> Tuple[Any, np.ndarray]:
    """
    Add points and bounding boxes to the SAM2 model for object segmentation.

    This function processes both points and bounding boxes, adding them to the SAM2 model
    for object segmentation. It handles multiple objects and can work with either points,
    bounding boxes, or both.

    Args:
        sam2 (nn.Module): The SAM2 model instance.
        inference_state (Any): The current inference state of the SAM2 model.
        points (np.ndarray): Array of points to add, shape (N, M, 2) where N is the number
                             of objects and M is the number of points per object.
        neg_points (np.ndarray): Array of negative points to add, shape (N, 2) where N is the number
                             of negative points.
        bboxes (np.ndarray): Array of bounding boxes to add, shape (K, 4) where K is the
                             number of bounding boxes.
        frame_idx (int): The frame index to interact with.
        obj_id (int, optional): The starting object ID. Defaults to 1.

    Returns:
        Tuple[Any, np.ndarray]: A tuple containing:
            - The updated inference state.
            - An array of mask logits for all processed objects.

    Note:
        - If points is None, no point processing will occur.
        - If bboxes is None, no bounding box processing will occur.
        - The function uses add_points_to_frame and add_bbox_to_frame internally.
    """
    out_mask_logits = []
    
    # Process points
    if points is not None or neg_points is not None:
        inference_state, _, mask_logits = add_points_to_frame(
            sam2=sam2,
            inference_state=inference_state,
            points=points,
            neg_points=neg_points,
            frame_idx=frame_idx,
            ann_obj_id=obj_id
        )
        out_mask_logits.extend(mask_logits)
    
    # Process bboxes
    if bboxes is not None:
        inference_state, _, mask_logits = add_bbox_to_frame(
            sam2=sam2,
            inference_state=inference_state,
            bbox=bboxes,
            frame_idx=frame_idx,
            ann_obj_id=obj_id
        )
        out_mask_logits.extend(mask_logits)
    
    return inference_state, out_mask_logits



def add_points_to_frame(
    sam2: nn.Module,
    inference_state: Any,
    points: np.ndarray = None,
    neg_points: np.ndarray = None,
    frame_idx: int = 0,
    ann_obj_id: int = 1,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Add points to a specific frame for object segmentation using SAM2.

    Args:
        sam2 (nn.Module): The SAM2 model instance.
        inference_state (Any): The current inference state.
        points (np.ndarray): Array of positive points to add, shape (N, 2) where N is the number of points.
        neg_points (np.ndarray): Array of negative points to add, shape (M, 2) where M is the number
                             of negative points.
        frame_idx (int, optional): The frame index to interact with. Defaults to 0.
        ann_obj_id (int, optional): Unique ID for the object being interacted with. Defaults to 1.

    Returns:
        Tuple: A tuple containing (inference_state, out_obj_ids, out_mask_logits).
    """
    # Initialize labels for positive points
    labels = np.ones(len(points), dtype=np.int32) if points is not None else np.asarray([], dtype=np.int32)
    neg_labels = np.zeros(len(neg_points), dtype=np.int32) if neg_points is not None else np.asarray([], dtype=np.int32)
    
    # Assure fitting dimensions
    points = points.reshape(-1, 2) if points is not None else np.empty((0, 2), dtype=np.int32)
    neg_points = neg_points.reshape(-1, 2) if neg_points is not None else np.empty((0, 2), dtype=np.int32)

    # Concatenate points and labels
    used_points = np.concatenate((points, neg_points), axis=0)
    used_labels = np.concatenate([labels, neg_labels], axis=0)
    
    _, out_obj_ids, out_mask_logits = sam2.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=ann_obj_id,
        points=used_points,
        labels=used_labels,
    )
    
    return inference_state, out_obj_ids, out_mask_logits



def add_bbox_to_frame(
    sam2: nn.Module,
    inference_state: Any,
    bbox: np.ndarray,
    frame_idx: int = 0,
    ann_obj_id: int = 1,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Add a bounding box to a specific frame for object segmentation using SAM2.

    Args:
        sam2 (nn.Module): The SAM2 model instance.
        inference_state (Any): The current inference state.
        bbox (np.ndarray): Array representing the bounding box, shape (4,) in format [x_min, y_min, x_max, y_max].
        frame_idx (int, optional): The frame index to interact with. Defaults to 0.
        ann_obj_id (int, optional): Unique ID for the object being interacted with. Defaults to 1.

    Returns:
        Tuple: A tuple containing (inference_state, out_obj_ids, out_mask_logits).
    """
    _, out_obj_ids, out_mask_logits = sam2.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=ann_obj_id,
        box=bbox,
    )
    
    return inference_state, out_obj_ids, out_mask_logits
