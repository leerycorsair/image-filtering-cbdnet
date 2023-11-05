
import cv2
import numpy as np


class Mean():
    def __init__(self) -> None:
        pass

    def predict_image(self, input_image: cv2.Mat) -> np.ndarray:
        output_image = cv2.blur(input_image, (5, 5))
        return output_image
