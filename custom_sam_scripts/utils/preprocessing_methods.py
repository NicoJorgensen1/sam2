import numpy as np
import cv2



def subtract_initial_frame(new_frame, initial_frame):
    return new_frame - initial_frame


def divide_by_initial_frame(new_frame, initial_frame):
    return (new_frame.astype(np.float32) / (initial_frame.astype(np.float32) + 1e-6)) * 255


def apply_low_high_pass(current_frame):
    low_pass = cv2.GaussianBlur(current_frame, (15, 15), 0)
    high_pass = current_frame.astype(np.float32) - low_pass.astype(np.float32)
    high_pass = np.clip(high_pass, 0, 255).astype(np.uint8)
    return low_pass, high_pass

def apply_clahe(frame, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Ensure the frame is 8-bit single-channel
    if frame.dtype != np.uint8:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Ensure the frame is single-channel
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(frame)


def scale_image(img) -> np.ndarray:
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


# Fourier Transform to Enhance Low-Frequency Changes
def low_freq_enhancement(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = image.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 30  # Radius to cut off high frequencies
    mask[crow-r:crow+r, ccol-r:ccol+r] = 1
    dft_shift = dft_shift * mask
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# Adaptive Thresholding
def adaptive_threshold(image, max_value=255):
    # Ensure the image is 8-bit single-channel
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Ensure the image is single-channel
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return cv2.adaptiveThreshold(image, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


# Difference of Gaussians (DoG)
def difference_of_gaussians(image, sigma1=1, sigma2=2):
    blur1 = cv2.GaussianBlur(image, (0, 0), sigma1)
    blur2 = cv2.GaussianBlur(image, (0, 0), sigma2)
    dog = blur1 - blur2
    return cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def inverting_images(img):
    return np.abs(255 - img).astype(np.uint8)


def gamma_transform(image, gamma, c=1):
    """
    Applies power-law (gamma) transformation to the input image.
    
    Args:
    - image: Input grayscale image (numpy array).
    - gamma: The gamma value for the transformation.
    - c: Scaling constant (default is 1).
    
    Returns:
    - Transformed image (numpy array).
    """
    # Normalize the image to the range 0-1
    normalized_img = image / 255.0
    
    # Apply the gamma correction
    gamma_corrected = c * np.power(normalized_img, gamma)
    
    # Rescale back to 0-255
    gamma_corrected = np.clip(gamma_corrected * 255, 0, 255).astype(np.uint8)
    
    return gamma_corrected





def histogram_equalization(image):
    """
    Applies histogram equalization to enhance the contrast of an image.
    
    Args:
    - image: Input grayscale image (numpy array).
    
    Returns:
    - Contrast-enhanced image (numpy array).
    """
    return cv2.equalizeHist(image)


