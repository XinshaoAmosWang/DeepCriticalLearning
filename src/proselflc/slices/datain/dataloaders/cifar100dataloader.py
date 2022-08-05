from functools import partial

import torch
from torch.utils.data import DataLoader

from proselflc.exceptions import ParamException
from proselflc.trainer.utils import intlabel2onehot

from ..datasets.cifar100dataset import CIFAR100Dataset
from ..transforms.cifar100transforms import (
    cifar100_transform_test_data,
    cifar100_transform_train_data,
)
from ..utils import BalancedBatchSampler, set_torch_seed


class CIFAR100DataLoader(DataLoader):
    """
    CIFAR100 Dataloader with customed settings.

    What is special here versus DataLoader:
        1. train is bool and required.
            1.1. which dataset
            1.2. shuffle=train accordingly.
            1.3. set data_transform and target_tranform accordingly.

    Args:
        train (bool, required):
            If true, it is a training dataloader.
            Otherwise, it is a testing dataloader
            1. shuffle(bool, not required):
                It is hidden in this class.
                being equeal to train (bool, required).
            2. which dataset will be set accordingly.
            3. transform will be set accordingly.
        num_workers:
            inherited from DataLoader.
        batch_size:
            inherited from DataLoader
    """

    # overwrite
    def __init__(
        self,
        params: dict = {
            "train": True,
            "num_workers": 4,
            "batch_size": 128,
            "symmetric_noise_rate": 0,
        },
    ) -> None:
        cifar100_transform_intlabel2onehot = partial(
            intlabel2onehot,
            class_num=params["num_classes"],
        )
        if params["sampler"] not in ["BalancedBatchSampler", None]:
            error_msg = "params['sampler']: {}, not in {}".format(
                params["sampler"],
                ["BalancedBatchSampler", None],
            )
            raise ParamException(error_msg)
        if params["train"] not in [True, False]:
            error_msg = "params['train']: {}, not in {}".format(
                params["train"],
                [True, False],
            )
            raise ParamException(error_msg)

        if params["train"]:
            self._dataset = CIFAR100Dataset(
                params,
                data_transform=cifar100_transform_train_data,
                target_transform=cifar100_transform_intlabel2onehot,
            )
        else:
            self._dataset = CIFAR100Dataset(
                params,
                data_transform=cifar100_transform_test_data,
                target_transform=cifar100_transform_intlabel2onehot,
            )

        if params["train"]:
            if "seed" in params.keys() and params["seed"] is not None:
                import random

                import numpy

                set_torch_seed(
                    numpy=numpy,
                    torch=torch,
                    random=random,
                    seed=params["seed"],
                )
                g = torch.Generator()
                g.manual_seed(params["seed"])

                def seed_worker(worker_id):
                    worker_seed = torch.initial_seed() % 2 ** 32
                    numpy.random.seed(worker_seed)
                    random.seed(worker_seed)

                #
                train_batch_sampler = BalancedBatchSampler(
                    labels=self._dataset.targets,
                    n_classes=params["classes_per_batch"],
                    n_samples=params["batch_size"] // params["classes_per_batch"],
                    seed=params["seed"],
                )
                super().__init__(
                    dataset=self._dataset,
                    num_workers=params["num_workers"],
                    batch_sampler=train_batch_sampler,
                    #
                    worker_init_fn=seed_worker,
                    generator=g,
                )
            else:
                super().__init__(
                    dataset=self._dataset,
                    shuffle=True,
                    num_workers=params["num_workers"],
                    batch_size=params["batch_size"],
                )
        else:
            # val or test data loaders
            super().__init__(
                dataset=self._dataset,
                shuffle=False,
                num_workers=params["num_workers"],
                batch_size=params["batch_size"] * 2,
            )
