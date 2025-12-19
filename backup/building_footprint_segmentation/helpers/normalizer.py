import numpy as np
import cv2


def divide_by_255(img: np.ndarray) -> np.ndarray:
    """

    :param img:
    :return:
    """
    return img / 255


def binary_label(mask: np.ndarray) -> np.ndarray:
    """

    :param mask:
    :return:
    """
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    #normalized_mask = divide_by_255(mask)
    normalized_mask = (mask>0).astype(np.uint8)
    if len(normalized_mask.shape) == 3:
        normalized_mask = normalized_mask[:,:,0]
    return np.expand_dims(normalized_mask, -1)

"""
def min_max_image_net(img: np.ndarray) -> np.ndarray:
   
   
    out = np.zeros_like(img).astype(np.float32)
    for i in range(img.shape[2]):
        c = img[:, :, i].min()
        d = img[:, :, i].max()

        t = (img[:, :, i] - c) / (d - c)
        out[:, :, i] = t
    out.astype(np.float32)
    out -= np.ones(out.shape) * (0.485, 0.456, 0.406)
    out /= np.ones(out.shape) * (0.229, 0.224, 0.225)
    return out
"""
def min_max_image_net(img):
    """
    Normalize image using ImageNet mean and std
    Input: uint8 image (0-255) in BGR or RGB format
    Output: float32 normalized image
    """
    
    # Convert to float and scale to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # ImageNet normalization (assuming RGB format)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Normalize
    img = (img - mean) / std
    
    return img


def divide_255_image_net(img: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32) / 255
    out -= np.ones(out.shape) * (0.485, 0.456, 0.406)
    out /= np.ones(out.shape) * (0.229, 0.224, 0.225)
    return out
