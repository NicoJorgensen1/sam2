from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import os




def plot_and_save_images(
        images: List[List[np.ndarray]],
        titles: List[List[str]],
        save_path: Optional[str] = None,
        show_plot: Optional[bool] = False
    ) -> None:
    """
    Plot a grid of images with corresponding titles and save the plot.
    
    Args:
    images (list of lists): A 2D list containing the images to be plotted.
    titles (list of lists): A 2D list containing the titles for each image.
    save_path (str): File path to save the output plot.
    show_plot (bool): Whether to show the plot.
    """
    # Assure the inputs are nested lists 
    if not isinstance(images[0], list):
        images = [images]
    if not isinstance(titles[0], list):
        titles = [titles]

    # Create a subplot with the same dimensions as the nested images list
    nrows = len(images)
    ncols = len(images[0])
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 18))
    if nrows==1:
        axs = axs[np.newaxis, :]
    if ncols==1:
        axs = axs[:, np.newaxis]

    # Loop through rows and columns
    for i in range(len(images)):
        for j in range(len(images[i])):
            axs[i, j].imshow(images[i][j], cmap='gray')
            axs[i, j].set_title(titles[i][j])
            axs[i, j].axis('off')

    fig.tight_layout()

    # Save the plot
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show(block=True)
    plt.close()
    return 
