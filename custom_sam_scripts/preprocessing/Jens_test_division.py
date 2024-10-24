import matplotlib
matplotlib.use('TkAgg')
from CameraAISharedLibrary.print_args_func import print_args
from CameraAISharedLibrary.str2bool_func import str2bool
from custom_sam_scripts.utils.drawing_utils.plot_save_images import plot_and_save_images
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
import os


def pixelwise_division(
        image1: np.ndarray,
        image2: np.ndarray
    ) -> np.ndarray:
    """Perform pixel-wise division of two images."""
    # Ensure the images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Error: Images must have the same dimensions for pixel-wise division.")

    # Avoid division by zero by adding a small epsilon to image2
    epsilon = 1e-10
    return np.divide(image1.astype(np.float32), image2.astype(np.float32) + epsilon)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize the image to the range [0, 255]."""
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm_image.astype(np.uint8)



def main(
        img1_path: str,
        img2_path: str,
        save_dir: str,
        show_plot: bool
    ) -> None:
    # Load images
    image1_color = cv2.cvtColor(cv2.imread(img1_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    image_1_gray = cv2.cvtColor(image1_color, cv2.COLOR_RGB2GRAY)
    image2_color = cv2.cvtColor(cv2.imread(img2_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    image_2_gray = cv2.cvtColor(image2_color, cv2.COLOR_RGB2GRAY)

    # Perform pixel-wise division
    img1_over_img2 = pixelwise_division(image_1_gray, image_2_gray)
    img2_over_img1 = pixelwise_division(image_2_gray, image_1_gray)

    # Normalize the resulting images
    img1_over_img2_normalized = normalize_image(img1_over_img2)
    img2_over_img1_normalized = normalize_image(img2_over_img1)

    # Define the images and titles
    images = [
        [image1_color, image2_color],
        [image_1_gray, image_2_gray],
        [img1_over_img2, img2_over_img1],
        [img1_over_img2_normalized, img2_over_img1_normalized]
    ]
    titles = [
        ['Original Image 1', 'Original Image 2'],
        ['Grayscale Image 1', 'Grayscale Image 2'],
        ['Image 1 / Image 2', 'Image 2 / Image 1'],
        ['Normalized (Image 1 / Image 2)', 'Normalized (Image 2 / Image 1)']
    ]

    # Call the plotting function with save_dir
    plot_and_save_images(images=images, titles=titles, save_dir=save_dir, show_plot=show_plot)




### Run from CLI 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform pixel-wise division on two images.")
    parser.add_argument("--img1_path", type=str, default="/Image_Datasets/InfuSight/ENCXYN/SHL01/image_1.jpeg", help="Path to the first image")
    parser.add_argument("--img2_path", type=str, default="/Image_Datasets/InfuSight/ENCXYN/SHL01/image_2.jpeg", help="Path to the second image")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the output plot")
    parser.add_argument("--show_plot", type=str2bool, default=False, help="Whether to show the plot")
    args = parser.parse_args()

    # Edit the input args 
    args.save_dir = os.path.dirname(args.img1_path) if "none" in str(args.save_dir).lower() else args.save_dir

    # Print the arguments
    print_args(args, init_str="This is the input args chosen when running Jens' preprocessing script")

    # Run the main function
    main(args.img1_path, args.img2_path, args.save_dir, args.show_plot)
