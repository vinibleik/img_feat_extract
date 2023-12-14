from typing import Literal

import cv2 as cv
import numpy as np

from features.feature import Feature


class FeatSIFT(Feature):
    def __init__(
        self,
        filepath: str,
        desc_out: str,
        img: np.ndarray | None = None,
        save_img_out: bool = False,
        img_out: str = "",
        ext: Literal["npy", "txt", "csv"] = "npy",
    ):
        super().__init__(filepath, desc_out, img, save_img_out, img_out, ext)
        self._feature_name = "SIFT"
        self.feature = cv.SIFT_create()
