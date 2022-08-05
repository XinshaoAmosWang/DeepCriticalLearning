import os
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
                "data_name": "cifar100",
                "num_classes": 100,  # 1000
                "device": "gpu",
                #
                "num_workers": 8,
                #
                "counter": "iteration",
                "classes_per_batch": 8,
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

        self.params["total_epochs"] = 80
        self.params["eval_interval"] = 400

        #
        #
        self.params["lr_scheduler"] = "WarmupMultiStepSchedule"
        self.params["sampler"] = "BalancedBatchSampler"

    def test_trainer_cifar100(self):
        k = 0

        for (
            self.params["symmetric_noise_rate"],
            (self.params["network_name"], self.params["weight_decay"]),
            #
            #
            (self.params["batch_size"], self.params["classes_per_batch"]),
            #
            self.params["momentum"],
            #
            self.params["warmup_epochs"],
            self.params["lr"],
            self.params["batch_accumu_steps"],
            #
            (self.params["milestones"], self.params["gamma"]),
            #
            (
                self.params["total_epochs"],
                self.params["eval_interval"],
            ),
            #
            self.params["loss_mode"],
            self.params["trust_mode"],
            #
            (
                self.params["loss_name"],
                self.params["logit_soften_T"],
            ),
            self.params["epsilon"],
        ) in product(
            [
                0.4,
                0.2,
                0.0,
                0.6,
            ],
            [
                ("resnet18", 2e-3),
                ("shufflenetv2", 1e-3),
            ],
            #
            #
            # 2022/03/04: (128, 8) converges much slower!!!
            [(128, 64)],  # fix this
            #
            [0.9],  # fix this
            #
            # last time, it was 8
            # will this be better?
            [0],  # warmup epochs
            [0.2],  # lr
            # sensitive or not?
            [10],  # batch accumu
            #
            # fix lr scheduler?
            [([20000, 30000], 0.1)],
            #
            # fix epoch
            [(100, 500)],
            [
                "cross entropy",
            ],
            [
                "global*(1-H(p)/H(u))",
                "global only",
                "global*max_p",
            ],
            #
            [
                ("crossentropy", None),
                ("labelsmoothing", None),
                #
                ("confidencepenalty", 1.0),
                ("confidencepenalty", 0.5),
                ("confidencepenalty", 0.8),
                ("confidencepenalty", 0.6),
                ("confidencepenalty", 0.4),
                #
                ("labelcorrection", 1.0),
                ("labelcorrection", 0.5),
                ("labelcorrection", 0.8),
                ("labelcorrection", 0.6),
                ("labelcorrection", 0.4),
            ],
            [0.5],
        ):
            k = k + 1
            print(k)

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


if __name__ == "__main__":

    work_dir = os.getenv(
        "SM_CHANNEL_WORK_DIR",
        "/home/xinshao/tpami_proselflc_experiments_calibration/",
    )
    TestTrainer.WORK_DIR = work_dir

    print(TestTrainer.WORK_DIR)

    unittest.main()
