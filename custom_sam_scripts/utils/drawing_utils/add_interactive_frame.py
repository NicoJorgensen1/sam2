from AITrainingSharedLibrary.set_display_env_var import set_display_env_var
set_display_env_var()
from custom_sam_scripts.utils.drawing_utils.show_stuff import show_points, show_box, show_mask
from custom_sam_scripts.utils.sam2_utils.add_points_or_bboxes_to_frame import add_points_and_bboxes
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import numpy as np




def interactive_frame_annotation(frame, sam2, inference_state):
    """
    Create an interactive annotation interface for a given frame.

    This function allows users to annotate a frame with positive points (left-click),
    negative points (right-click), and bounding boxes (middle-click and drag).
    It updates the SAM2 model after each annotation.

    Args:
        frame (numpy.ndarray): The input frame to be annotated.
        sam2 (nn.Module): The SAM2 model instance.
        inference_state (Any): The current inference state of the SAM2 model.

    Returns:
        tuple: A tuple containing three numpy arrays:
            - points (numpy.ndarray): Array of positive point coordinates, shape (N, 2).
            - neg_points (numpy.ndarray): Array of negative point coordinates, shape (M, 2).
            - bboxes (numpy.ndarray): Array of bounding box coordinates, shape (K, 4).

    Usage:
        The function opens a matplotlib window where the user can:
        - Left-click to add positive points (green stars)
        - Right-click to add negative points (red stars)
        - Middle-click and drag to create bounding boxes (green rectangles)
        After each annotation, it updates the SAM2 model and displays the current mask.

    Note:
        The function will block execution until the user closes the matplotlib window.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(frame, cmap='gray')
    
    points = []
    neg_points = []
    bboxes = []
    
    def update_sam2():
        nonlocal inference_state
        inference_state, mask_logits = add_points_and_bboxes(
            sam2, inference_state, 
            np.array(points), np.array(bboxes), np.array(neg_points)
        )
        if mask_logits:
            ax.clear()
            ax.imshow(frame, cmap='gray')
            show_mask(mask=mask_logits[-1], ax=ax)
            show_points(coords=np.array(points), labels=np.ones(len(points)), ax=ax)
            show_points(coords=np.array(neg_points), labels=np.zeros(len(neg_points)), ax=ax)
            for bbox in bboxes:
                show_box(box=bbox, ax=ax)
        fig.canvas.draw()

    def onclick(event):
        if event.button == 1:  # Left click
            points.append([event.xdata, event.ydata])
        elif event.button == 3:  # Right click
            neg_points.append([event.xdata, event.ydata])
        update_sam2()

    def line_select_callback(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        bboxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        update_sam2()

    rs = RectangleSelector(ax, line_select_callback, useblit=True,
                           button=[2],  # Middle mouse button
                           minspanx=5, minspany=5,
                           spancoords='pixels',
                           interactive=True)
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("Left click: positive point, Right click: negative point, Middle click and drag: bounding box")
    plt.show()
    
    return np.array(points), np.array(neg_points), np.array(bboxes)
