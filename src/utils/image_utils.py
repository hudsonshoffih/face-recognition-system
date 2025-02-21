import cv2
import numpy as np

def resize_image(image, width=None, height=None):
    if width and height:
        return cv2.resize(image, (width, height))
    elif width:
        ratio = width / image.shape[1]
        return cv2.resize(image, (width, int(image.shape[0] * ratio)))
    elif height:
        ratio = height / image.shape[0]
        return cv2.resize(image, (int(image.shape[1] * ratio), height))
    return image