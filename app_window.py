import os
from PyQt5 import QtWidgets
import cv2
import urllib.request
from interface import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from methods.my_cnn import MyCNN
from methods.median import Median
from methods.mean import Mean
from methods.bilateral import Bilateral
from methods.gaussian import Gaussian
from utils.common import read_img, write_img, psnr, ssim
from datetime import datetime
import numpy as np
import time

from dataclasses import dataclass
from consts import PATHS, NAMES


@dataclass
class Image:
    data: np.ndarray
    name: str


@dataclass
class Metric:
    method: str
    psnr_value: float
    ssim_value: float


class AppWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(AppWindow, self).__init__()
        self.original_img = None
        self.edited_img = None
        self.ui_setup()

    def ui_setup(self) -> None:
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.bind_buttons()
        self.show()

    def bind_buttons(self):
        self.ui.input_button.clicked.connect(self.set_input)
        self.ui.ideal_button.clicked.connect(self.set_ideal)
        self.ui.output_button.clicked.connect(self.set_output)
        self.ui.calc_button.clicked.connect(self.calc_perform)

    def set_ideal(self) -> None:
        try:
            filename = QFileDialog.getOpenFileName(
                self, "Выбор файла", "C:\\", "Изображения (*.jpg *.png *.jpeg)"
            )[0]
            self.ui.ideal_line_edit.setText(filename)
        except:
            QMessageBox.warning(self, "Ошибка", "Некорректный файл")

    def set_input(self) -> None:
        try:
            filename = QFileDialog.getOpenFileName(
                self, "Выбор файла", "C:\\", "Изображения (*.jpg *.png *.jpeg)"
            )[0]
            self.ui.input_line_edit.setText(filename)
        except:
            QMessageBox.warning(self, "Ошибка", "Некорректный файл")

    def set_output(self) -> None:
        try:
            directoryname = QFileDialog.getExistingDirectory(self, "Выбор директории", "C:\\")
            self.ui.output_line_edit.setText(directoryname)
        except:
            QMessageBox.warning(self, "Ошибка", "Некорректный путь")

    def calc_perform(self) -> None:
        images = list()
        metrics = list()
        ideal_image = None

        modes = {0: "cpu", 1: "gpu"}
        mode = modes[self.ui.mode_combobox.currentIndex()]
        try:
            input_file = self.ui.input_line_edit.text()
            output_directory = self.ui.output_line_edit.text()
            if self.ui.metrics_checkbox.isChecked() and self.ui.ideal_line_edit.text() != "":
                ideal_file = self.ui.ideal_line_edit.text()
                ideal_image = read_img(ideal_file)
                images.append(Image(ideal_image, "ideal.png"))
            if output_directory == "Путь по умолчанию":
                output_directory = PATHS["default_save"]
            output_folder = self.create_output_folder(output_directory)

            input_image = read_img(input_file)
            try:
                m = MyCNN(PATHS["models"], NAMES["my_cnn"], mode)
                output_image = m.predict_image(input_image)
            except FileNotFoundError:
                QMessageBox.warning(self, "Ошибка", "Отсутствует предобученная модель")
                return

            images.append(Image(input_image, "noised.png"))
            images.append(Image(output_image, "denoised_mycnn.png"))
            if ideal_image is not None:
                metrics.append(
                    Metric("my_cnn", psnr(ideal_image, output_image), ssim(ideal_image, output_image))
                )
        except:
            QMessageBox.warning(self, "Ошибка", "Некорректный файл изображения")
            return

        if self.ui.mean_checkbox.isChecked():
            m = Mean()
            output_image = m.predict_image(input_image)
            images.append(Image(output_image, "denoised_mean.png"))
            if ideal_image is not None:
                metrics.append(
                    Metric("mean", psnr(ideal_image, output_image), ssim(ideal_image, output_image))
                )

        if self.ui.gaussian_checkbox.isChecked():
            m = Gaussian()
            output_image = m.predict_image(input_image)
            images.append(Image(output_image, "denoised_gaussian.png"))
            if ideal_image is not None:
                metrics.append(
                    Metric(
                        "gaussian", psnr(ideal_image, output_image), ssim(ideal_image, output_image)
                    )
                )

        if self.ui.median_checkbox.isChecked():
            m = Median()
            output_image = m.predict_image(input_image)
            images.append(Image(output_image, "denoised_median.png"))
            if ideal_image is not None:
                metrics.append(
                    Metric(
                        "median", psnr(ideal_image, output_image), ssim(ideal_image, output_image)
                    )
                )

        if self.ui.bilateral_checkbox.isChecked():
            m = Bilateral()
            output_image = m.predict_image(input_image)
            images.append(Image(output_image, "denoised_bilateral.png"))
            if ideal_image is not None:
                metrics.append(
                    Metric(
                        "bilateral",
                        psnr(ideal_image, output_image),
                        ssim(ideal_image, output_image),
                    )
                )
        if self.ui.scunet_checkbox.isChecked():
            os.environ["REPLICATE_API_TOKEN"] = "r8_0o46KLISICN0zxOgMDuIrmhIO48qxzw3W4Vl3"
            import replicate

            response = replicate.run(
                model_version="cszn/scunet:df9a3c1dbc6c1f7f4c2d244f68dffa2699a169cf5e701e0d6a009bf6ff507f26",
                input={"image": open(input_file, "rb")},
            )

            output_image_url = response["denoised_image"]
            req = urllib.request.urlopen(output_image_url)
            output_image = cv2.imdecode(np.asarray(bytearray(req.read()), dtype=np.uint8), -1)

            images.append(Image(output_image, "denoised_scunet.png"))
            if ideal_image is not None:
                metrics.append(
                    Metric(
                        "scunet", psnr(ideal_image, output_image), ssim(ideal_image, output_image)
                    )
                )

        self.save_results(images, output_folder, metrics)

    def save_results(self, images: list[Image], output_folder: str, metrics: list[Metric]) -> None:
        for image in images:
            write_img(image.data, image.name, output_folder)

        if self.ui.metrics_checkbox.isChecked():
            f = open(output_folder + "report.txt", "a")
            for metric in metrics:
                f.write(
                    "Method = {:<10} PSNR = {:<10.3f} SSIM = {:<10.3f}\n".format(
                        metric.method, metric.psnr_value, metric.ssim_value
                    )
                )
            f.close()

    def create_output_folder(self, output_directory: str) -> str:
        time_label = str(datetime.utcnow())
        for char in " -.:":
            time_label = time_label.replace(char, "_")
        output_folder = output_directory + time_label
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        return output_folder + "/"
