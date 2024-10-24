import numpy as np
import cv2




def select_largest_component(mask):
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))

    # Find sizes of all connected components (except background which is 0)
    sizes = np.bincount(labels.flatten())[1:]

    # Find the label of the largest component
    largest_component_label = np.argmax(sizes) + 1  # Adding 1 because background is 0

    # Create a new mask with only the largest component
    return (labels == largest_component_label).astype(np.uint8) 
