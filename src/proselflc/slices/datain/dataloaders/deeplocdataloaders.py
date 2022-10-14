from functools import partial

from torch.utils.data import DataLoader

from proselflc.exceptions import ParamException
from proselflc.slices.datain.datasets.deeplocdatasets import DeepLocDataset
from proselflc.slices.datain.utils import BalancedBatchSampler, set_torch_seed
from proselflc.trainer.utils import intlabel2onehot

# from transformers import Trainer, TrainingArguments
# from proselflc.slices.networks.transformers.prot_bert_bfd_seqlevel import (
#     prot_bert_bfd_seqclassifier,
# )

# def deeploc_traindataloader(
#     params: dict = {
#         "output_dir": "./results",
#         "batch_size": 8,
#         #
#         "compute_metrics": compute_metrics,
#     }
# ):
#     if "compute_metrics" not in params.keys():
#         params["compute_metrics"] = compute_metrics
#     if "seed" not in params.keys():
#         params["seed"] = 123
#
#     training_args = TrainingArguments(
#         output_dir=params["output_dir"],  # output directory
#         per_device_train_batch_size=params[
#             "batch_size"
#         ],  # batch size per device during training
#         per_device_eval_batch_size=params["batch_size"],  # batch size for evaluation
#         seed=params["seed"],  # Seed for experiment reproducibility
#     )
#
#     tfmer_trainer = Trainer(
#         model=prot_bert_bfd_seqclassifier(
#             # Note here, the only reason to use tfmer_trainer is to
#             # use its get_train_dataloader(),
#             # there I will change params
#             # to reduce resources consumption
#             {
#                 "network_name": params["network_name"],
#                 "num_hidden_layers": 1,
#                 "num_attention_heads": 1,
#                 #
#                 "num_classes": params["num_classes"],
#             },
#         ),  # the instantiated 🤗 Transformers model to be trained
#         compute_metrics=params["compute_metrics"],  # evaluation metrics
#         #
#         # for generating train loader
#         train_dataset=DeepLocDataset(
#             split="train",
#             tokenizer_name=params["network_name"],
#             max_length=params["max_seq_length"],
#             task_name=params["task_name"],
#         ),
#         #
#         # args mainly for batch size
#         args=training_args,
#     )
#
#     return tfmer_trainer.get_train_dataloader()


def DeepLocTrainLoaderTestset(
    params: dict = {
        "split": None,
        "batch_size": 128,
        "task_name": "MS",
    },
):
    if (params["task_name"], params["num_classes"]) not in zip(
        ("MS-with-unknown", "MS", "SubcellularLoc"), (2, 2, 10)
    ):
        error_msg = "task_name={}, num_classes={}, not in {}".format(
            params["task_name"],
            params["num_classes"],
            #
            [pair for pair in zip(("MS", "SubcellularLoc"), (2, 10))],
        )
        raise ParamException("non-matched task name and num_classes: " + error_msg)

    tokenizer_name = params["network_name"]
    transform_intlabel2onehot = partial(
        intlabel2onehot,
        class_num=params["num_classes"],
    )

    if params["split"] == "train":
        # return deeploc_traindataloader(
        #     params=params,
        # )
        train_dataset = DeepLocDataset(
            split="train",
            tokenizer_name=tokenizer_name,
            max_length=params["max_seq_length"],
            task_name=params["task_name"],
            target_transform=transform_intlabel2onehot,
            seed=params["seed"],
        )
        train_batch_sampler = BalancedBatchSampler(
            labels=train_dataset.labels,
            n_classes=params["classes_per_batch"],
            n_samples=params["batch_size"] // params["classes_per_batch"],
            seed=params["seed"],
        )

        # operation to keep deterministic reproducibility
        import random

        import numpy
        import torch

        set_torch_seed(
            torch=torch,
            seed=params["seed"],
        )
        g = torch.Generator()
        g.manual_seed(params["seed"])

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(
            dataset=train_dataset,
            num_workers=params["num_workers"],
            #
            batch_sampler=train_batch_sampler,
            #
            worker_init_fn=seed_worker,
            generator=g,
        )
    elif params["split"] in ["valid", "val", "test"]:
        return DeepLocDataset(
            split="valid or test: the same dataset",
            tokenizer_name=tokenizer_name,
            max_length=params["max_seq_length"],
            task_name=params["task_name"],
            target_transform=transform_intlabel2onehot,
            seed=params["seed"],
        )
    else:
        error_msg = (
            "incorrect dataset config for DeepLocTrainLoaderTestset: "
            "split={}, not in {}".format(
                params["split"], ["train", "val", "valid", "test"]
            )
        )
        raise ParamException(error_msg)
