import os
import re
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# NOTE
# val set = test set
from proselflc.exceptions import ParamException
from proselflc.slices.datain.utils import set_torch_seed


class DeepLocDataset(Dataset):
    """Deep Loc dataset:
    - task name = MS, class_num = 2
    - task name = SubcellularLoc, class_num = 10
    """

    def __init__(
        self,
        split="train",
        tokenizer_name="Rostlab/prot_bert_bfd",
        max_length=1534,  # max_length of test seqs
        # 2 for MS, 10 for location
        task_name="MS",  # or "SubcellularLoc"
        target_transform: Optional[Callable] = None,
        seed: int = None,
    ):
        if seed is not None:
            set_torch_seed(
                numpy=np,
                seed=seed,
            )

        if tokenizer_name == "Rostlab_prot_bert_bfd_seq":
            tokenizer_name = "Rostlab/prot_bert_bfd"
        self.datasetFolderPath = "dataset/"
        self.trainFilePath = os.path.join(
            self.datasetFolderPath, "deeploc_per_protein_train.csv"
        )
        self.testFilePath = os.path.join(
            self.datasetFolderPath, "deeploc_per_protein_test.csv"
        )

        self.downloadDeeplocDataset()

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, do_lower_case=False
        )
        self.target_transform = target_transform

        self.max_length = max_length
        if split == "train":
            self.seqs, self.labels = self.load_dataset(
                path=self.trainFilePath,
                task_name=task_name,
                split=split,
            )
        else:
            self.seqs, self.labels = self.load_dataset(
                path=self.testFilePath,
                task_name=task_name,
                split=split,
                drop_longer_than_max=True,
            )
        self.tokenize_seqs()

    def downloadDeeplocDataset(self):
        deeplocDatasetTrainUrl = (
            "https://www.dropbox.com/s/"
            "vgdqcl4vzqm9as0/deeploc_per_protein_train.csv?dl=1"
        )
        deeplocDatasetValidUrl = (
            "https://www.dropbox.com/s/"
            "jfzuokrym7nflkp/deeploc_per_protein_test.csv?dl=1"
        )

        if not os.path.exists(self.datasetFolderPath):
            os.makedirs(self.datasetFolderPath)

        def download_file(url, filename):
            response = requests.get(url, stream=True)
            with tqdm.wrapattr(
                open(filename, "wb"),
                "write",
                miniters=1,
                total=int(response.headers.get("content-length", 0)),
                desc=filename,
            ) as fout:
                for chunk in response.iter_content(chunk_size=4096):
                    fout.write(chunk)

        if not os.path.exists(self.trainFilePath):
            download_file(deeplocDatasetTrainUrl, self.trainFilePath)

        if not os.path.exists(self.testFilePath):
            download_file(deeplocDatasetValidUrl, self.testFilePath)

    def load_dataset(self, path, task_name, drop_longer_than_max=False, split=None):
        if task_name not in ["MS-with-unknown", "MS", "SubcellularLoc"]:
            error_msg = "incorrect task name: " "task_name={}, not in {}".format(
                task_name,
                ["MS-with-unknown", "MS", "SubcellularLoc"],
            )
            raise ParamException(error_msg)
        df = pd.read_csv(path, names=["input", "loc", "membrane"], skiprows=1)
        if task_name == "MS-with-unknown":
            # MS with unknown labels
            if split == "train":
                unique_class_names = np.unique(df["membrane"])
                print("unique_class_names: {}".format(unique_class_names))

                #
                self.labels_dic = {0: "Soluble", 1: "Membrane"}
                # known labels
                df["labels"] = [0] * len(df["input"])
                df.loc["M" == df["membrane"], ["labels"]] = 1

                # generate random labels 0 or 1 for unknown labels
                df.loc["U" == df["membrane"], ["labels"]] = np.random.randint(
                    0, 2, size=np.sum("U" == df["membrane"])
                )
                #
                #
            else:
                # filtering the datset: df["membrane"].isin(["M", "S"])
                df = df.loc[df["membrane"].isin(["M", "S"])]
                df = df.reset_index(drop=True)
                self.labels_dic = {0: "Soluble", 1: "Membrane"}
                df["labels"] = np.where(df["membrane"] == "M", 1, 0)
        elif task_name == "MS":
            # labels for MS
            # filtering the dataset: df["membrane"].isin(["M", "S"])
            df = df.loc[df["membrane"].isin(["M", "S"])]
            df = df.reset_index(drop=True)
            self.labels_dic = {0: "Soluble", 1: "Membrane"}
            df["labels"] = np.where(df["membrane"] == "M", 1, 0)
        else:
            # labels for SubcellularLoc
            unique_class_names = np.unique(df["loc"])
            print("unique_class_names: {}".format(len(unique_class_names)))

            self.labels_dic = {}
            df["labels"] = [0] * len(df["input"])
            for class_idx, class_name in enumerate(unique_class_names):
                self.labels_dic[class_idx] = class_name
                df.loc[class_name == df["loc"], ["labels"]] = class_idx

        from dataprep.eda import create_report

        cfg = {"wordcloud.enable": False}
        df["input_seq_length"] = [
            len(re.sub(r"[UZOB]", "X", "".join(df.loc[idx]["input"].split())))
            for idx in range(df.shape[0])
        ]
        loaded_df_report = create_report(df, config=cfg)
        sink_path = "dataset/" + "/deeploc_eda_reports/{}".format(
            task_name + "_" + split
        )
        loaded_df_report.save(
            sink_path,
        )
        print()

        if drop_longer_than_max:
            # True for non-train data
            self.non_drop_marks = []
            for idx in range(df.shape[0]):
                seq = "".join(df.loc[idx]["input"].split())
                seq = re.sub(r"[UZOB]", "X", seq)
                seq_len = len(seq)
                self.non_drop_marks.append(seq_len <= self.max_length)
            df = df.loc[self.non_drop_marks]
            df = df.reset_index(drop=True)

        # return
        seq_list = list(df["input"])
        label_list = list(df["labels"])
        assert len(seq_list) == len(label_list)
        return seq_list, label_list

    def tokenize_seqs(self):
        self.seq_inputidswithlabels = []

        # tokenization one by one
        for idx in range(len(self.labels)):
            # Make sure there is a space between every token,
            # and map rarely amino acids
            seq = " ".join("".join(self.seqs[idx].split()))
            seq = re.sub(r"[UZOB]", "X", seq)

            # every seq will be truncated or padded to a fix length.
            # this max length is for number of tokens, therefore + 2
            # to include start and end tokens
            seq_ids = self.tokenizer(
                seq,
                truncation=True,
                padding="max_length",
                max_length=self.max_length + 2,
            )

            sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
            if self.target_transform is not None:
                sample["labels"] = torch.tensor(self.target_transform(self.labels[idx]))
            else:
                sample["labels"] = torch.tensor(self.labels[idx])

            self.seq_inputidswithlabels.append(sample)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.seq_inputidswithlabels[idx]


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # weighted take into consideration "label imbalance"
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
