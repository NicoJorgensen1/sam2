import matplotlib.pyplot as plt
import numpy as np



def show_mask(mask, ax, obj_id=None, random_color=False):
    """
    Display a mask on a given matplotlib axis.

    Args:
        mask (np.ndarray): The mask to be displayed.
        ax (matplotlib.axes.Axes): The axis on which to display the mask.
        obj_id (int, optional): Object ID for color selection. If None, uses default color. Defaults to None.
        random_color (bool, optional): If True, uses a random color. Defaults to False.

    The function applies a color to the mask and displays it on the given axis.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    """
    Display points on a given matplotlib axis.

    Args:
        coords (np.ndarray): Array of point coordinates, shape (N, 2).
        labels (np.ndarray): Array of point labels (0 for negative, 1 for positive).
        ax (matplotlib.axes.Axes): The axis on which to display the points.
        marker_size (int, optional): Size of the markers. Defaults to 200.

    The function plots positive points as green stars and negative points as red stars.
    """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    """
    Display a bounding box on a given matplotlib axis.

    Args:
        box (list or np.ndarray): Bounding box coordinates [x0, y0, x1, y1].
        ax (matplotlib.axes.Axes): The axis on which to display the box.

    The function adds a green rectangle representing the bounding box to the given axis.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

