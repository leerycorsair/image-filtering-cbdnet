
import numpy as np

import cv2


class Median():
    def __init__(self) -> None:
        pass

    def predict_image(self, input_image: cv2.Mat) -> np.ndarray:
        output_image = cv2.medianBlur(input_image, 5)
        return output_image
