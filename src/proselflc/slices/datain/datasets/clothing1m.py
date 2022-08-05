import os
from typing import Callable, Optional

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from proselflc.slices.datain.utils import set_torch_seed

# adapted according to
# https://github.com/pxiangwu/PLC/blob/master/clothing1m/data_clothing1m.py


class Clothing1M(data.Dataset):
    def __init__(
        self,
        params={
            "data_root": None,
            "split": "train",  # "val", or "test",
            "cls_size": 18976,  # balanced samples per class
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
        self.cls_size = params["cls_size"]

        self.train = False
        self.data_transform = data_transform
        self.target_transform = target_transform

        if self.split == "train":
            self.train = True
            file_path = os.path.join(
                self.data_root, "annotations/noisy_train_key_list.txt"
            )
            label_path = os.path.join(self.data_root, "annotations/my_train_label.txt")
        elif self.split == "val":
            file_path = os.path.join(
                self.data_root, "annotations/clean_val_key_list.txt"
            )
            label_path = os.path.join(self.data_root, "annotations/my_val_label.txt")
        else:
            file_path = os.path.join(
                self.data_root, "annotations/clean_test_key_list.txt"
            )
            label_path = os.path.join(self.data_root, "annotations/my_test_label.txt")

        with open(file_path) as fid:
            image_list = [line.strip() for line in fid.readlines()]

        with open(label_path) as fid:
            label_list = [int(line.strip()) for line in fid.readlines()]

        if self.split == "train" and self.cls_size is not None:
            # balanced sampling !!!!
            # TODO: be cautious and provide discussion about this

            self.image_list = np.array(image_list)
            self.label_list = np.array(label_list)

            np_labels = np.array(self.label_list)
            x = np.unique(np_labels)

            res_img_list = []
            res_label_list = []

            for i in x:
                idx = np.where(np_labels == i)[0]
                idx = np.random.permutation(idx)
                idx = idx[: self.cls_size]

                res_img_list.append(self.image_list[idx])
                res_label_list.append(self.label_list[idx])

            self.image_list = np.concatenate(res_img_list).tolist()
            self.label_list = np.concatenate(res_label_list).tolist()
        else:
            self.image_list = image_list
            self.label_list = label_list

        self.targets = self.label_list  # this is for backward code compatibility

    def __getitem__(self, index):
        image_file_name = self.image_list[index]
        image_path = os.path.join(self.data_root, image_file_name)
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
        return len(self.label_list)

    def make_weights_for_balanced_classes(self):
        nclasses = np.max(np.array(self.label_list)) + 1
        count = [0] * nclasses
        weight_per_class = [0.0] * nclasses

        for target in self.label_list:
            count[np.array(target).astype(np.int64)] += 1

        for i in range(nclasses):
            weight_per_class[i] = float(self.__len__()) / float(count[i])

        self.weight_per_sample_list = [0] * self.__len__()
        for idx in range(self.__len__()):
            target = self.label_list[idx]
            self.weight_per_sample_list[idx] = weight_per_class[
                np.array(target).astype(np.int64)
            ]
        return torch.DoubleTensor(self.weight_per_sample_list)
