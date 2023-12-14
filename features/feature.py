import os
from typing import Literal

import cv2 as cv
import numpy as np


class Feature:
    def __init__(
        self,
        filepath: str,
        desc_out: str,
        img: np.ndarray | None = None,
        save_img_out: bool = False,
        img_out: str = "",
        ext: Literal["npy", "txt", "csv"] = "npy",
    ):
        self.filepath = filepath
        self.__basename = Feature.get_base_name(self.filepath)

        self.desc_out = desc_out

        self.img = img or self.__gen_img()
        self.gray = self.__gen_gray_img()

        self.save_img_out = save_img_out
        if self.save_img_out and img_out == "":
            raise ValueError(
                "parameter img_out must be provided when save_img is settled to True!"
            )
        self.img_out = img_out
        self.ext = ext

        self._feature_name = "Feature"
        self.feature = None
        self.kp: tuple | None = None
        self.desc: np.ndarray | None = None

    @staticmethod
    def get_base_name(filepath: str) -> str:
        basename = os.path.basename(filepath)
        *names, _ = basename.split(".")
        return ".".join(names)

    @staticmethod
    def save_txt(filepath: str, arr: np.ndarray):
        np.savetxt(filepath, arr)

    @staticmethod
    def save_csv(filepath: str, arr: np.ndarray):
        np.savetxt(filepath, arr, delimiter=",")

    @staticmethod
    def save_npy(filepath: str, arr: np.ndarray):
        np.save(filepath, arr)

    def detect(self) -> tuple:
        if self.feature is None:
            raise ValueError("Feature must be initiated!")
        self.kp = self.feature.detect(self.gray, None)
        return self.kp

    def compute(self) -> tuple[tuple, np.ndarray]:
        if self.feature is None:
            raise ValueError("Feature must be initiated!")
        self.kp, self.desc = self.feature.compute(self.gray, self.kp)
        return (self.kp, self.desc)

    def extract(self) -> tuple[tuple, np.ndarray]:
        self.detect()
        return self.compute()

    def __gen_img(self) -> np.ndarray:
        return cv.imread(self.filepath)

    def __gen_gray_img(self) -> np.ndarray:
        return cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

    def save_img(self) -> str:
        if not self.save_img_out:
            return f"{self.save_img=} not saving image"

        if self.kp is None:
            raise ValueError("Keypoints must be initiated!")

        save_dir = os.path.join(self.img_out, self._feature_name)
        save_name = f"{self._feature_name.lower()}_keypoints_{self.__basename}.jpg"
        save_path = os.path.join(save_dir, save_name)
        os.makedirs(save_dir, exist_ok=True)
        img_out = cv.drawKeypoints(
            self.gray,
            self.kp,
            self.img,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv.imwrite(save_path, img_out)
        return save_path

    def save_desc(self) -> str:
        if self.desc is None:
            raise ValueError("Descriptors must be initiated!")

        save_dir = os.path.join(self.desc_out, self._feature_name)
        save_name = (
            f"{self._feature_name.lower()}_descriptors_{self.__basename}.{self.ext}"
        )
        save_path = os.path.join(save_dir, save_name)
        os.makedirs(save_dir, exist_ok=True)
        getattr(Feature, f"save_{self.ext}")(save_path, self.desc)
        return save_path

    def save(self) -> tuple[str, str]:
        return (self.save_desc(), self.save_img())

    def pipeline(self) -> tuple[str, str]:
        self.extract()
        return self.save()
