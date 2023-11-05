import numpy as np
import cv2
from skimage import color
from skimage.metrics import structural_similarity


def read_img(filename: str) -> np.ndarray:
    img = cv2.imread(filename)
    return img


def write_img(img: np.ndarray, filename: str, path: str):
    cv2.imwrite(path + filename, img)


def hwc_to_chw(img: np.ndarray) -> np.ndarray:
    return np.transpose(img, axes=[2, 0, 1]).astype("float32")


def chw_to_hwc(img: np.ndarray) -> np.ndarray:
    return np.transpose(img, axes=[1, 2, 0]).astype("float32")


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    img1_gray = color.rgb2gray(img1)
    img2_gray = color.rgb2gray(img2)
    ssim_score = structural_similarity(
        img1_gray, img2_gray, data_range=img1_gray.max() - img1_gray.min()
    )
    return ssim_score


def split_image(image, patch_size=256):
    height, width = image.shape[:2]

    num_patches_x = (width + patch_size - 1) // patch_size
    num_patches_y = (height + patch_size - 1) // patch_size

    new_width = num_patches_x * patch_size
    new_height = num_patches_y * patch_size

    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_image[:height, :width, :] = image

    patches = np.array(
        [
            new_image[
                j * patch_size : j * patch_size + patch_size,
                i * patch_size : i * patch_size + patch_size,
            ]
            for j in range(num_patches_y)
            for i in range(num_patches_x)
        ]
    )

    return patches


def restore_image(patches, width, height, patch_size=256):
    num_patches_x = (width + patch_size - 1) // patch_size
    num_patches_y = (height + patch_size - 1) // patch_size
    restored_width = num_patches_x * patch_size
    restored_height = num_patches_y * patch_size

    restored_image = np.zeros((restored_height, restored_width, 3), dtype=np.uint8)

    for i, patch in enumerate(patches):
        row = i // num_patches_x
        col = i % num_patches_x
        left = col * patch_size
        upper = row * patch_size
        restored_image[upper : upper + patch_size, left : left + patch_size] = patch

    restored_image = restored_image[: height, : width]

    return restored_image
