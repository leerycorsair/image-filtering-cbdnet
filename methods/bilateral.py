

import numpy as np

import cv2


class Bilateral():
    def __init__(self) -> None:
        pass

    def predict_image(self, input_image: cv2.Mat) -> np.ndarray:
        output_image = cv2.bilateralFilter(input_image, 5, 120, 120)
        return output_image
