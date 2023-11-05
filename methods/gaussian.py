

import numpy as np

import cv2


class Gaussian():
    def __init__(self) -> None:
        pass

    def predict_image(self, input_image: cv2.Mat) -> np.ndarray:
        output_image = cv2.GaussianBlur(input_image, (5, 5), 1.5)
        return output_image
