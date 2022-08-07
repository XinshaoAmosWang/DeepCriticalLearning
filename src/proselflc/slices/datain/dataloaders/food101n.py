from functools import partial

import torch
from torch.utils.data import DataLoader

from proselflc.exceptions import ParamException
from proselflc.trainer.utils import intlabel2onehot

from ..datasets.food101n import Food101N
from ..transforms.clothing1mtransforms import (
    clothing1m_transform_test_resizecrop,
    clothing1m_transform_test_resizeonly,
    clothing1m_transform_train_rc,
    clothing1m_transform_train_rrcsr,
)
from ..utils import BalancedBatchSampler, set_torch_seed


class Food101NLoader(DataLoader):
    """
    What is special here versus DataLoader:
        1. split: train, or test.
            1.1. which dataset
            1.2. shuffle=train accordingly.
            1.3. set data_transform and target_tranform accordingly.

    Args:
        split:  train, or test (str, required):
            1. shuffle(bool, not required):
                It is hidden in this class.
                set by split value
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
            "data_root": None,
            "split": "train",
            "num_workers": 4,
            "batch_size": 128,
            "test_transform": "resizecrop",
        },
    ) -> None:
        target_transform_intlabel2onehot = partial(
            intlabel2onehot,
            class_num=params["num_classes"],
        )
        # Sanity checks
        if params["split"] not in ["train", "test"]:
            error_msg = "params['split']: {}, not in {}".format(
                params["split"],
                ["train", "test"],
            )
            raise ParamException(error_msg)
        if params["test_transform"] not in ["resizeonly", "resizecrop"]:
            error_msg = "params['test_transform']: {}, not in {}".format(
                params["test_transform"],
                ["resizeonly", "resizecrop"],
            )
            raise ParamException(error_msg)

        if params["train_transform"] not in ["train_rrcsr", "train_rc"]:
            error_msg = "params['train_transform']: {}, not in {}".format(
                params["train_transform"],
                ["train_rrcsr", "train_rc"],
            )
            raise ParamException(error_msg)
        if params["sampler"] not in ["BalancedBatchSampler"]:
            error_msg = "params['sampler']: {}, not in {}".format(
                params["sampler"],
                ["BalancedBatchSampler"],
            )
            raise ParamException(error_msg)

        if params["split"] == "train":
            if params["train_transform"] == "train_rrcsr":
                self._dataset = Food101N(
                    params,
                    data_transform=clothing1m_transform_train_rrcsr(params),
                    target_transform=target_transform_intlabel2onehot,
                )
            else:
                self._dataset = Food101N(
                    params,
                    data_transform=clothing1m_transform_train_rc,
                    target_transform=target_transform_intlabel2onehot,
                )
        # testing data transform
        elif params["test_transform"] == "resizeonly":
            self._dataset = Food101N(
                params,
                data_transform=clothing1m_transform_test_resizeonly,
                target_transform=target_transform_intlabel2onehot,
            )
        elif params["test_transform"] == "resizecrop":
            self._dataset = Food101N(
                params,
                data_transform=clothing1m_transform_test_resizecrop,
                target_transform=target_transform_intlabel2onehot,
            )
        else:
            error_msg = "incorrect dataset config for Food101N: {}".format(
                params,
            )
            raise ParamException(error_msg)

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

        if params["split"] == "train":
            # only support BalancedBatchSampler
            train_batch_sampler = BalancedBatchSampler(
                labels=self._dataset.label_list,
                n_classes=params["classes_per_batch"],
                n_samples=params["batch_size"] // params["classes_per_batch"],
                seed=params["seed"],
            )
            if "seed" in params.keys() and params["seed"] is not None:
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
                    num_workers=params["num_workers"],
                    batch_sampler=train_batch_sampler,
                )
        else:
            # i.e., test data loaders
            if "seed" in params.keys() and params["seed"] is not None:
                super().__init__(
                    dataset=self._dataset,
                    shuffle=(params["split"] == "train"),  # only if train, shuffle.
                    num_workers=params["num_workers"],
                    batch_size=params["batch_size"] * 2,
                    #
                    worker_init_fn=seed_worker,
                    generator=g,
                )
            else:
                super().__init__(
                    dataset=self._dataset,
                    shuffle=(params["split"] == "train"),  # only if train, shuffle.
                    num_workers=params["num_workers"],
                    batch_size=params["batch_size"] * 2,
                )
