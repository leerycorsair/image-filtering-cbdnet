import torch
import torch.nn as nn
import os
from methods.models import my_cnn_model
import numpy as np
from utils import common
import cv2


class MyCNN:
    def __init__(
        self,
        path: str,
        name: str,
        mode: str,
    ) -> None:
        self.mode = mode
        if self.mode == "cpu":
            self.model = my_cnn_model.MyCNNModel()
            self.model = nn.DataParallel(self.model)
            self.model.eval()

            if os.path.exists(os.path.join(path, name)):
                model_info = torch.load(os.path.join(path, name), map_location=torch.device("cpu"))
                self.model.load_state_dict(model_info["state_dict"])
            else:
                raise FileNotFoundError
        else:
            self.model = my_cnn_model.MyCNNModel()
            self.model = nn.DataParallel(self.model)
            self.model.eval()

            if os.path.exists(os.path.join(path, name)):
                model_info = torch.load(os.path.join(path, name))
                self.model.load_state_dict(model_info["state_dict"])
            else:
                raise FileNotFoundError

    def predict_image(self, input_image: cv2.Mat) -> np.ndarray:
        input_patches = common.split_image(input_image)
        output_patches = []
        for patch in input_patches:
            output_patches.append(self._predict_patch(patch))
        output_image = common.restore_image(
            output_patches, input_image.shape[1], input_image.shape[0]
        )
        return output_image

    def _predict_patch(self, input_patch: cv2.Mat) -> np.ndarray:
        input_patch = input_patch[:, :, ::-1] / 255.0
        input_patch = np.array(input_patch).astype("float32")
        if self.mode == "cpu":
            input_var = torch.from_numpy(common.hwc_to_chw(input_patch)).unsqueeze(0).cpu()
        else:
            input_var = torch.from_numpy(common.hwc_to_chw(input_patch)).unsqueeze(0).cuda()

        with torch.no_grad():
            output = self.model(input_var)

        if self.mode == "cpu":
            output_patch = common.chw_to_hwc(output[0, ...].cpu().numpy())
        else:
            output_patch = common.chw_to_hwc(output[0, ...].cuda().numpy())

        output_patch = np.uint8(np.round(np.clip(output_patch, 0, 1) * 255.0))[:, :, ::-1]
        return output_patch
