import os
from typing import Callable, Optional

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from proselflc.slices.datain.utils import set_torch_seed

# https://github.com/kuangliu/pytorch-retinanet/blob/master/transform.py
# TODO: transform: figure out resize keeping the ratio.


class Food101N(data.Dataset):
    def __init__(
        self,
        params={
            "data_root": None,
            "split": "train",  # "val", or "test",
        },
        data_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        # It is fine to always force reproducibility
        # for dataset, without input dependence.
        if "seed" in params.keys() and params["seed"] is not None:
            set_torch_seed(
                numpy=np,
                torch=torch,
                os=os,
                seed=params["seed"],
            )
        # TODO: to add sanity checks
        self.data_root = params["data_root"]
        self.split = params["split"]

        self.train = False
        self.data_transform = data_transform
        self.target_transform = target_transform

        if self.split == "train":
            self.train = True
            self.image_list = np.load(os.path.join(self.data_root, "train_images.npy"))
            self.targets = np.load(os.path.join(self.data_root, "train_targets.npy"))
        else:
            self.train = False
            self.image_list = np.load(os.path.join(self.data_root, "test_images.npy"))
            self.targets = np.load(os.path.join(self.data_root, "test_targets.npy"))
        self.label_list = list(self.targets)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path)
        image = image.convert("RGB")

        target = self.label_list[index]
        target = np.array(target).astype(np.int64)

        if self.data_transform is not None:
            image = self.data_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.targets)
