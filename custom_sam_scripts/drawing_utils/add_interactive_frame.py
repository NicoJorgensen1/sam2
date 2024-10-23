import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np




def interactive_frame_annotation(frame):
    """
    Create an interactive annotation interface for a given frame.

    This function allows users to annotate a frame with positive points (left-click),
    negative points (right-click), and bounding boxes (middle-click and drag).

    Args:
        frame (numpy.ndarray): The input frame to be annotated.

    Returns:
        tuple: A tuple containing three numpy arrays:
            - points (numpy.ndarray): Array of positive point coordinates, shape (N, 2).
            - neg_points (numpy.ndarray): Array of negative point coordinates, shape (M, 2).
            - bboxes (numpy.ndarray): Array of bounding box coordinates, shape (K, 4).

    Usage:
        The function opens a matplotlib window where the user can:
        - Left-click to add positive points (red stars)
        - Right-click to add negative points (blue stars)
        - Middle-click and drag to create bounding boxes

    Note:
        The function will block execution until the user closes the matplotlib window.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(frame, cmap='gray')
    
    points = []
    neg_points = []
    bboxes = []
    
    def onclick(event):
        if event.button == 1:  # Left click
            points.append([event.xdata, event.ydata])
            ax.plot(event.xdata, event.ydata, 'r*', markersize=10)
        elif event.button == 3:  # Right click
            neg_points.append([event.xdata, event.ydata])
            ax.plot(event.xdata, event.ydata, 'b*', markersize=10)
        fig.canvas.draw()
    
    def line_select_callback(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        bboxes.append([x1, y1, x2, y2])
    
    rs = RectangleSelector(ax, line_select_callback, useblit=True,
                           button=[1, 3],  # Left and right mouse buttons
                           minspanx=5, minspany=5,
                           spancoords='pixels',
                           interactive=True)
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("Left click: positive point, Right click: negative point, Middle click and drag: bounding box")
    plt.show()
    
    return np.array(points), np.array(neg_points), np.array(bboxes)
