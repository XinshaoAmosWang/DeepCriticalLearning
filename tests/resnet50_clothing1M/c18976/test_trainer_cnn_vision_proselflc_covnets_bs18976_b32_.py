import os
import random
import time
import unittest
from itertools import product

import pandas
import torch

from proselflc.optim.sgd_multistep import SGDMultiStep
from proselflc.trainer.trainer_cnn_vision_derivedgrad import Trainer


class TestTrainer(unittest.TestCase):
    WORK_DIR = None

    def set_params_vision(self):
        self.params.update(
            {
                "data_name": "clothing1m_withbs",
                "data_root": "/home/xinshao/"
                "tpami_proselflc_experiments_calibration/"
                "input_dir/clothing1m",
                #
                # fix now: balanced sampling=>NO, not dropping examples
                "cls_size": 18976,  # balanced samples per class, reduce
                #
                "device": "gpu",
                #
                "num_workers": 16,
                #
                "counter": "iteration",
                #
                "seed": 123,
            }
        )

    def setUp(self):
        """
        This function is an init for all tests
        """
        self.params = {}
        self.set_params_vision()

        self.params["network_name"] = "resnet50_tv"
        self.params["pretrained"] = True
        self.params["symmetric_noise_rate"] = 0.0  # placeholder only
        #
        #
        self.params["train_transform"] = "train_rrcsr"
        self.params["lr_scheduler"] = "WarmupMultiStepSchedule"
        self.params["sampler"] = "BalancedBatchSampler"

    def test_trainer_cifar100(self):
        total = 1
        single_split = total // 1
        my_exp_list = list(range(1, total + 1))
        random.Random(self.params["seed"]).shuffle(my_exp_list)
        #
        k = 0
        split = 0

        for (
            self.params["cls_size"],
            self.params["num_classes"],
            self.params["layer_split_lr_scale"],
            (self.params["batch_size"], self.params["classes_per_batch"]),
            #
            self.params["momentum"],
            #
            self.params["transit_time_ratio"],
            (self.params["lr"], self.params["weight_decay"]),
            self.params["batch_accumu_steps"],
            #
            (
                self.params["dropout"],
                #
                self.params["milestones"],
                self.params["gamma"],
                self.params["total_epochs"],
                self.params["eval_interval"],
            ),
            self.params["warmup_epochs"],
            #
            (self.params["min_scale"], self.params["rotation"]),
            self.params["freeze_bn"],
            #
            self.params["loss_mode"],
            self.params["trust_mode"],
            #
            #
            (
                self.params["loss_name"],
                self.params["logit_soften_T"],
                self.params["epsilon"],
                self.params["exp_base"],
            ),
        ) in product(
            [18976],
            [
                15,  # one prototype for the other classes
            ],
            [
                [-2, 2],
            ],  # fix this
            [
                (32, 8),
            ],  # fix this
            #
            [0.9],  # fix momentum
            [0.5],  # fix transit_time_ratio
            #
            [
                # (self.params["lr"], self.params["weight_decay"])
                (0.01, 2e-2),
            ],
            [
                # batch_accumu_steps
                8,
            ],
            #
            # lr scheduler
            [
                # dropout, gamma
                (0.15, [5000, 8500, 9200, 11700], 0.2, 2, 100),
            ],
            [
                # warmup
                0.18,
            ],
            #
            # data augmentation
            [
                (0.75, 15),
            ],  # fix this
            [False],  # freeze_bn
            #
            #
            [
                "cross entropy",
            ],
            [
                "global*max_p",
            ],
            #
            # < 1, entropy decreases
            [
                ("proselflc", 0.42, None, 12),
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

            self.params["layer_split_lr_scale"] = str(
                self.params["layer_split_lr_scale"]
            )
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
