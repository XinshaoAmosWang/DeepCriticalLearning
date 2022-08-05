import os
import random
import time
import unittest
from itertools import product

import pandas
import torch

from proselflc.optim.sgd_multistep import SGDMultiStep
from proselflc.trainer.trainer_cnn_vision_derivedgrad_adaptedfordeeploc import Trainer

"""
{
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.0,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.0,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "max_position_embeddings": 40000,
  "num_attention_heads": 16,
  "num_hidden_layers": 30,
  "type_vocab_size": 2,
  "vocab_size": 30
}
"""


class TestTrainer(unittest.TestCase):
    WORK_DIR = None

    def set_params(self):
        self.params.update(
            {
                "data_name": "deeploc_prottrans",
                "num_classes": 2,
                "task_name": "MS",
                #
                "device": "gpu",
                "num_workers": 1,
                #
                # network
                "network_name": "Rostlab_prot_bert_bfd_seq",
                "num_hidden_layers": 6,
                "num_attention_heads": 16,
                #
                "counter": "iteration",
                #
                "batch_size": 16,
                "classes_per_batch": 2,  # at most 2
                #
                "max_seq_length": 434,  # smaller -> more noise
                #
                #
                "seed": 123,
            }
        )

    def setUp(self):
        """
        This function is an init for all tests
        """
        self.params = {}
        self.set_params()

        self.params["loss_name"] = "proselflc"
        #
        self.params["symmetric_noise_rate"] = 0.0  # placeholder only
        #
        #
        self.params["lr_scheduler"] = "WarmupMultiStepSchedule"
        self.params["sampler"] = "BalancedBatchSampler"

    def test_trainer(self):
        total = 96
        single_split = total // 8
        my_exp_list = list(range(1, total + 1))
        random.Random(self.params["seed"]).shuffle(my_exp_list)
        #
        k = 0
        split = 0

        for (
            (
                self.params["num_classes"],
                self.params["task_name"],
                self.params["lr"],
                (self.params["milestones"], self.params["gamma"]),
            ),
            #
            #
            (self.params["batch_size"], self.params["classes_per_batch"]),
            #
            self.params["momentum"],
            self.params["weight_decay"],
            #
            self.params["transit_time_ratio"],
            self.params["warmup_epochs"],
            self.params["batch_accumu_steps"],
            #
            #
            (
                self.params["total_epochs"],
                self.params["eval_interval"],
            ),
            #
            self.params["loss_mode"],
            #
            (
                self.params["loss_name"],
                self.params["exp_base"],
                self.params["logit_soften_T"],
                self.params["trust_mode"],
            ),
        ) in product(
            [
                (2, "MS-with-unknown", 0.01, ([5000], 0.1)),
            ],
            #
            #
            # batch size, and classes per batch
            [(32, 2)],  # fix this
            #
            [0.9],  # fix this
            # weight decay is very sensitive
            # larger since I am using batch accu?
            # last time, it was 2e-3
            # will this be better?
            [1e-4],  # fix this
            #
            # transit ratio is very sensitive
            [0.50],  # transit ratio
            # last time, it was 8
            # will this be better?
            [0],  # warmup epochs
            # sensitive or not?
            [10],  # batch accumu
            #
            #
            # fix epoch
            [(40, 100)],
            [
                "cross entropy",
            ],
            #
            #
            # < 1, entropy decreases
            [
                ("proselflc", 12, 0.2, "global*(1-H(p)/H(u))"),
            ],
        ):
            k = k + 1
            print(k)
            if k not in my_exp_list[split * single_split : (split + 1) * single_split]:
                continue

            dt_string = time.strftime("%Y%m%d-%H%M%S")
            summary_writer_dir = (
                "{:0>3}_".format(k)
                + self.params["loss_name"]
                + "_warm"
                + str(self.params["warmup_epochs"])
                # + "_gamma"
                # + str(self.params["gamma"])
                # + "_"
                # + str(self.params["counter"])
                # + "_epo"
                # + str(self.params["total_epochs"])
                # + "_lr"
                # + str(self.params["lr"])
                # + "_"
                # + str(self.params["milestones"])
                + "_"
                + dt_string
            )
            self.params["summary_writer_dir"] = (
                self.WORK_DIR
                + "/"
                + self.params["data_name"]
                + "_symmetric_noise_rate_"
                + str(self.params["symmetric_noise_rate"])
                + "/"
                + self.params["network_name"]
                + "/"
                + summary_writer_dir
            )
            if not os.path.exists(self.params["summary_writer_dir"]):
                os.makedirs(self.params["summary_writer_dir"])

            trainer = Trainer(params=self.params)
            self.assertTrue(isinstance(trainer, Trainer))
            self.assertTrue(isinstance(trainer.optim, SGDMultiStep))

            self.params["milestones"] = str(self.params["milestones"])
            self.dataframe = pandas.DataFrame(self.params, index=[0])
            self.dataframe.to_csv(
                self.params["summary_writer_dir"] + "/params.csv",
                encoding="utf-8",
                index=False,
                sep="\t",
                mode="w",  #
            )

            # some more test
            trainer.train()
            torch.save(
                trainer.network,
                self.params["summary_writer_dir"] + "/model.pt",
            )
            del trainer
            torch.cuda.empty_cache()


if __name__ == "__main__":

    work_dir = os.getenv(
        "SM_CHANNEL_WORK_DIR",
        "/home/xinshao/tpami_proselflc_experiments_calibration/",
    )
    TestTrainer.WORK_DIR = work_dir

    print(TestTrainer.WORK_DIR)

    unittest.main()
