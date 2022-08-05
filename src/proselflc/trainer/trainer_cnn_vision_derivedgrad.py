import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import torch
from torch.utils.data import DataLoader

from proselflc.exceptions import ParamException
from proselflc.optim.sgd_multistep import SGDMultiStep
from proselflc.slicegetter.get_dataloader import DataLoaderPool
from proselflc.slicegetter.get_lossfunction import LossPool
from proselflc.slicegetter.get_network import NetworkPool
from proselflc.slices.datain.utils import set_torch_seed
from proselflc.trainer.utils import logits2probs_softmax

get_network = NetworkPool.get_network
get_dataloader = DataLoaderPool.get_dataloader
get_lossfunction = LossPool.get_lossfunction
colorscale = [[0, "#4d004c"], [0.5, "#f2e5ff"], [1, "#ffffff"]]


class Trainer:
    """

    Inputs:
        1. dataloader
        2. network with train mode: network.train(mode=True)
        3. loss
        4. optimiser
        5. device = cpu or gpu

    Functionality:
        1. build the graph according to params
            dataloader,
            network,
            loss function,
            optimiser
        2. batch training through dataloader, which is iterable.
        3.
    """

    def __init__(self, params):
        if "device" not in params.keys() or params["device"] not in ["cpu", "gpu"]:
            error_msg = (
                "The input params have no key of device. "
                + "params["
                + "device"
                + "] "
                + " has to be provided as cpu or gpu."
            )
            raise (ParamException(error_msg))
        self.device = params["device"]

        if "seed" in params.keys() and params["seed"] is not None:
            set_torch_seed(
                torch=torch,
                seed=params["seed"],
            )

        # network
        self.network_name = params["network_name"]
        self.network = get_network(params)
        self.network.train(mode=True)
        if self.device == "gpu":
            self.network = self.network.cuda()

        # dataloader
        self.data_name = params["data_name"]
        params["train"] = True
        if self.data_name in ["clothing1m_withbs", "food101n"]:
            params["split"] = "train"
            # only for place holder
            params["test_transform"] = "resizecrop"
            #
            self.traindataloader = get_dataloader(params)
        else:
            self.traindataloader = get_dataloader(params)

        self.total_epochs = params["total_epochs"]
        # time tracker
        self.loss_name = params["loss_name"]
        self.cur_time = 0
        self.counter = params["counter"]
        if self.counter == "iteration":
            # affected by batch size.
            params["total_iterations"] = self.total_epochs * len(self.traindataloader)

        # loss function
        self.loss_criterion = get_lossfunction(params)

        # TODO: create a getter for all optional optimisers
        # optim with optimser and lr scheduler
        self.warmup_epochs = params["warmup_epochs"]
        params["warmup_iterations"] = int(
            len(self.traindataloader) * self.warmup_epochs
        )
        self.warmup_iterations = params["warmup_iterations"]
        self.optim = SGDMultiStep(net_params=self.network.parameters(), params=params)

        # logit temperature scaling
        if "logit_soften_T" in params.keys() and params["logit_soften_T"] is not None:
            self.logit_soften_T = torch.tensor(
                params["logit_soften_T"],
                requires_grad=False,
            )
        else:
            self.logit_soften_T = None

        # logging misc ######################################
        # add summary writer
        self.summarydir = params["summary_writer_dir"]
        self.params = params
        self.noisy_data_analysis_prep()
        self.init_logger()
        # logging misc ######################################

    def noisy_data_analysis_prep(self):
        # special case for label noise
        self.cleantraindataloader = None
        if "symmetric_noise_rate" in self.params.keys():
            sym_noisy_key = "symmetric_noise_rate"
        elif "asymmetric_noise_rate_finea2b" in self.params.keys():
            sym_noisy_key = "asymmetric_noise_rate_finea2b"
        if self.params[sym_noisy_key] > 0.0:
            # to get clean train data
            self.params["train"] = True
            self.noise_rate = self.params[sym_noisy_key]
            self.params[sym_noisy_key] = 0.0
            self.cleantraindataloader = get_dataloader(self.params)
            self.params[sym_noisy_key] = self.noise_rate

            mask_list = np.array(
                self.cleantraindataloader._dataset.targets
            ) == np.array(self.traindataloader._dataset.targets)

            # clean and noisy subsets
            clean_indexes = [list[0] for list in np.argwhere(mask_list)]
            noisy_indexes = [list[0] for list in np.argwhere(np.invert(mask_list))]
            clean_subset = torch.utils.data.Subset(
                self.traindataloader._dataset,
                clean_indexes,
            )
            noisy_subset = torch.utils.data.Subset(
                self.traindataloader._dataset,
                noisy_indexes,
            )
            cleaned_noisy_subset = torch.utils.data.Subset(
                self.cleantraindataloader._dataset,
                noisy_indexes,
            )
            self.clean_subloader = DataLoader(
                dataset=clean_subset,
                shuffle=False,
                num_workers=self.params["num_workers"],
                batch_size=self.params["batch_size"] * 2,
            )
            self.noisy_subloader = DataLoader(
                dataset=noisy_subset,
                shuffle=False,
                num_workers=self.params["num_workers"],
                batch_size=self.params["batch_size"] * 2,
            )
            self.cleaned_noisy_subloader = DataLoader(
                dataset=cleaned_noisy_subset,
                shuffle=False,
                num_workers=self.params["num_workers"],
                batch_size=self.params["batch_size"] * 2,
            )

    def init_logger(self):
        self.accuracy_dynamics = {"epoch": []}
        self.loss_dynamics = {"epoch": []}
        self.entropy_dynamics = {"epoch": []}
        self.max_p_dynamics = {"epoch": []}

        self.metadata_dynamics = {"epoch": []}
        self.metadata_dynamics["lr_change_ratio"] = []
        self.metadata_dynamics["batch_mean_epsilon"] = []
        self.metadata_dynamics["batch_valid_ratio"] = []
        self.metadata_dynamics["batch_mean_gtrust"] = []
        self.metadata_dynamics["batch_mean_etrust"] = []
        self.batch_mean_epsilon = [-1, -1, -1, -1]
        #
        self.dataloaders = {}
        if self.cleantraindataloader is None:
            if self.data_name == "clothing1m_withbs":
                # resizecrop
                self.params["test_transform"] = "resizecrop"
                # self.params["split"] = "val"
                # self.valdataloader_resizecrop = get_dataloader(self.params)
                self.params["split"] = "test"
                self.testdataloader_resizecrop = get_dataloader(self.params)

                self.dataloaders = {
                    # As I am using iterwise evaluation.
                    # and batch sampler.
                    # "noisy_train": DataLoader(
                    #     dataset=self.traindataloader.dataset,
                    #     shuffle=False,
                    #     num_workers=self.params["num_workers"],
                    #     batch_size=self.params["batch_size"] * 2,
                    # ),
                    # "val_resizecrop": self.valdataloader_resizecrop,
                    "test_resizecrop": self.testdataloader_resizecrop,
                }
            elif self.data_name == "food101n":
                # resizecrop
                self.params["test_transform"] = "resizecrop"
                self.params["split"] = "test"
                self.testdataloader_resizecrop = get_dataloader(self.params)

                self.dataloaders = {
                    # As I am using iterwise evaluation.
                    # and batch sampler.
                    # "noisy_train": DataLoader(
                    #     dataset=self.traindataloader.dataset,
                    #     shuffle=False,
                    #     num_workers=self.params["num_workers"],
                    #     batch_size=self.params["batch_size"] * 2,
                    # ),
                    "test_resizecrop": self.testdataloader_resizecrop,
                }
            else:
                self.params["train"] = False
                self.testdataloader = get_dataloader(self.params)
                self.dataloaders = {
                    "clean_train": DataLoader(
                        dataset=self.traindataloader._dataset,
                        shuffle=False,
                        num_workers=self.params["num_workers"],
                        batch_size=self.params["batch_size"] * 2,
                    ),
                    "clean_test": self.testdataloader,
                }
        else:
            # noisy data
            self.params["train"] = False
            self.testdataloader = get_dataloader(self.params)

            self.dataloaders = {
                # "clean_train": self.cleantraindataloader,
                "clean_test": self.testdataloader,
                # "noisy_train": self.traindataloader,
                "noisy_subset": self.noisy_subloader,
                "clean_subset": self.clean_subloader,
                "cleaned_noisy_subset": self.cleaned_noisy_subloader,
            }
        for name in self.dataloaders.keys():
            self.accuracy_dynamics[name] = []
            self.loss_dynamics[name] = []
            self.entropy_dynamics[name] = []
            self.max_p_dynamics[name] = []

    def train(self) -> None:
        # #############################
        for epoch in range(1, self.total_epochs + 1):
            # train one epoch
            self.train_one_epoch(
                epoch=epoch,
                dataloader=self.traindataloader,
            )
        # #############################
        self.sink_csv_figures()

    def train_one_epoch(self, epoch: int, dataloader) -> None:
        self.network.train()  # self.network.train(mode=True)

        # reset gradients tensors
        self.optim.optimizer.zero_grad()
        self.optim.optimizer.step()
        #
        for batch_index, (raw_inputs, labels) in enumerate(dataloader):
            # #############################
            # track time
            if self.counter == "epoch":
                self.cur_time = epoch
            else:
                # epoch counter to iteration counter
                self.cur_time = (epoch - 1) * len(dataloader) + batch_index + 1
            # #############################

            # #############################
            # data ingestion
            network_inputs = raw_inputs
            if self.device == "gpu":
                network_inputs = network_inputs.cuda()
                labels = labels.cuda()
            # #############################

            # #############################
            # forward
            logits = self.network(network_inputs)
            if self.logit_soften_T is None:
                pred_probs = logits2probs_softmax(logits=logits)
            else:
                pred_probs = logits2probs_softmax(logits=logits)
                calibrated_pred_probs = logits2probs_softmax(
                    logits=logits.detach() / self.logit_soften_T,
                )
            # #############################

            # #############################
            # loss
            if self.loss_name != "proselflc":
                if self.logit_soften_T is None:
                    loss = self.loss_criterion(
                        pred_probs=pred_probs,
                        target_probs=labels,
                    )
                else:
                    loss = self.loss_criterion(
                        pred_probs=pred_probs,
                        target_probs=labels,
                        # only for confidence/entropy calibration
                        # tune logit_soften_T based on validation
                        calibrated_pred_probs=calibrated_pred_probs,
                    )
            else:
                loss = self.loss_criterion(
                    pred_probs=pred_probs,
                    target_probs=labels,
                    cur_time=self.cur_time,
                    batch_mean_epsilon=self.batch_mean_epsilon,
                    # only for confidence/entropy calibration
                    # tune logit_soften_T based on validation
                    calibrated_pred_probs=calibrated_pred_probs,
                )
            # #############################

            # #############################
            # function to extract grad
            # def set_grad(var):
            #     def hook(grad):
            #         var.grad = grad
            #
            #     return hook
            #
            # #register_hook for logits
            # logits.register_hook(set_grad(logits))

            # backward
            # Normalize our loss (if averaged)
            loss = loss / self.params["batch_accumu_steps"]

            if self.params["loss_name"] == "dm_exp_pi":
                # ########################################################
                # Implementation for derivative manipulation + Improved MAE
                # Novelty: From Loss Design to Derivative Design
                # Our work inspired: ICML-2020 (Normalised Loss Functions)
                # and ICML-2021 (Asymmetric Loss Functions)
                # ########################################################
                # remove orignal weights
                p_i = pred_probs[labels.nonzero(as_tuple=True)][:, None]
                logit_grad_derived = (pred_probs - labels) / (2.0 * (1.0 - p_i) + 1e-8)
                # add new weight: derivative manipulation
                logit_grad_derived *= (
                    torch.exp(
                        self.params["dm_beta"]
                        * (1.0 - p_i)
                        * torch.pow(p_i + 1e-8, self.params["dm_lambda"])
                    )
                    + 1e-8
                )
                # derivative normalisation,
                # which inspired the ICML-2020 paper-Normalised Loss Functions
                sum_weight = sum(
                    torch.exp(
                        self.params["dm_beta"]
                        * (1.0 - p_i)
                        * torch.pow(p_i + 1e-8, self.params["dm_lambda"])
                    )
                    + 1e-8
                )
                logit_grad_derived /= sum_weight
                logits.backward(logit_grad_derived)

            elif self.params["loss_name"] == "dm_standard_mae":
                p_i = pred_probs[labels.nonzero(as_tuple=True)][:, None]
                logit_grad_derived = (pred_probs - labels) * p_i
                logit_grad_derived /= self.params["batch_size"]
                logits.backward(logit_grad_derived)
            elif self.params["loss_name"] == "dm_standard_mse":
                p_i = pred_probs[labels.nonzero(as_tuple=True)][:, None]
                logit_grad_derived = (pred_probs - labels) * (-2.0 * p_i) * (p_i - 1)
                logit_grad_derived /= self.params["batch_size"]
                logits.backward(logit_grad_derived)
            elif self.params["loss_name"] == "dm_gce":
                p_i = pred_probs[labels.nonzero(as_tuple=True)][:, None]
                logit_grad_derived = (pred_probs - labels) * torch.pow(
                    p_i, self.params["dm_gce_q"]
                )
                logit_grad_derived /= self.params["batch_size"]
                logits.backward(logit_grad_derived)
            elif self.params["loss_name"] == "dm_sce":
                p_i = pred_probs[labels.nonzero(as_tuple=True)][:, None]
                logit_grad_derived = (pred_probs - labels) * self.params[
                    "dm_sce_mu"
                ] - 0.5 * self.params["dm_sce_Av"] * (pred_probs - labels) * (
                    -2.0 * p_i
                ) * (
                    p_i - 1
                )
                logit_grad_derived /= self.params["batch_size"]
                logits.backward(logit_grad_derived)
            else:
                loss.backward()

            # print(str(logits.requires_grad))
            # print(logits.shape)
            # print(pred_probs.shape)
            # print(torch.argmax(labels, 1)[0])
            # print(pred_probs[0, torch.argmax(labels, 1)[0]])
            # print(logits.grad[0,torch.argmax(labels, 1)[0]])
            # print(pred_probs[0,:])
            # print(sum(pred_probs[0,:]))
            # print(logits.grad[0, :])
            # print(sum(logits.grad.abs()[0, :]))

            # for logging, I am still using epoch, only to reduce changes here.
            if self.cur_time % self.params["eval_interval"] == 0:
                self.eval_helper(epoch)
                self.network.train()  # self.network.train(mode=True)

            # update params
            if self.cur_time % self.params["batch_accumu_steps"] == 0:
                # wait for multiple backward steps before zero_grad
                # https://github.com/jeonsworld/ViT-pytorch/blob/main/utils/scheduler.py
                # https://github.com/jeonsworld/ViT-pytorch/blob/main/train.py
                self.optim.lr_scheduler.step()

                self.optim.optimizer.step()
                self.optim.optimizer.zero_grad()

        #
        # final testing
        if (
            epoch == self.total_epochs
            and self.cur_time != self.params["total_iterations"]
        ):
            self.eval_helper(epoch)

    def eval_helper(self, epoch):
        print("Evaluating Network.....")
        print(
            "Iteration= ({},{})/({}, {}), lr={:.4f}, "
            "batch_mean_epsilon={:.4f}, valid batch size={:.4f}, "
            "batch_mean_gtrust={:.4f}, batch_mean_etrust={:.4f}".format(
                self.cur_time,
                epoch,
                self.params["total_iterations"],
                self.total_epochs,
                #
                self.optim.lr_scheduler.get_last_lr()[0],
                self.batch_mean_epsilon[0],
                #
                self.batch_mean_epsilon[1],
                #
                self.batch_mean_epsilon[2],
                #
                self.batch_mean_epsilon[3],
            )
        )
        self.metadata_dynamics["lr_change_ratio"].append(
            self.optim.lr_scheduler.get_last_lr()[0] / self.params["lr"]
        )
        self.metadata_dynamics["batch_mean_epsilon"].append(self.batch_mean_epsilon[0])
        self.metadata_dynamics["batch_valid_ratio"].append(
            self.batch_mean_epsilon[1] / self.params["batch_size"]
        )
        self.metadata_dynamics["batch_mean_gtrust"].append(self.batch_mean_epsilon[2])
        self.metadata_dynamics["batch_mean_etrust"].append(self.batch_mean_epsilon[3])
        self.metadata_dynamics["epoch"].append(self.cur_time)
        self.loss_dynamics["epoch"].append(self.cur_time)
        self.accuracy_dynamics["epoch"].append(self.cur_time)
        self.entropy_dynamics["epoch"].append(self.cur_time)
        self.max_p_dynamics["epoch"].append(self.cur_time)

        for eval_dataname, eval_dataloader in self.dataloaders.items():
            (eval_loss, eval_accuracy, eval_entropy, eval_max_p) = self.evaluation(
                dataloader=eval_dataloader,
            )
            self.loss_dynamics[eval_dataname].append(eval_loss)
            self.accuracy_dynamics[eval_dataname].append(eval_accuracy)
            self.entropy_dynamics[eval_dataname].append(eval_entropy)
            self.max_p_dynamics[eval_dataname].append(eval_max_p)
            print(
                eval_dataname + ": Loss= {:.4f}, Accuracy= {:.4f}, "
                "Entropy= {:.4f}, Max_p= {:.4f},".format(
                    eval_loss,
                    eval_accuracy,
                    eval_entropy,
                    eval_max_p,
                )
            )

    @torch.no_grad()
    def evaluation(self, dataloader):
        self.network.eval()

        test_loss = 0.0
        test_correct = 0.0
        test_entropy = 0.0
        test_max_p = 0.0

        for iter_idx, (raw_inputs, labels) in enumerate(dataloader):
            # #############################
            # data ingestion
            network_inputs = raw_inputs
            if self.device == "gpu":
                network_inputs = network_inputs.cuda()
                labels = labels.cuda()
            # #############################

            # #############################
            # forward
            logits = self.network(network_inputs)
            pred_probs = logits2probs_softmax(logits=logits)
            # #############################

            # #############################
            # loss
            cross_entropy = get_lossfunction({"loss_name": "crossentropy"})
            loss = cross_entropy(
                pred_probs=pred_probs,
                target_probs=labels,
            )
            # #############################

            test_loss += loss.item()
            _, preds = pred_probs.max(1)
            _, annotations = labels.max(1)
            test_correct += preds.eq(annotations).sum()

            # entropy calculation
            H_pred_probs = torch.sum(-pred_probs * torch.log(pred_probs + 1e-8), 1)
            H_uniform = -torch.log(torch.tensor(1.0 / self.params["num_classes"]))
            max_p, _indexes = torch.max(pred_probs, 1)
            test_entropy += torch.sum(H_pred_probs / H_uniform).item()
            test_max_p += torch.sum(max_p).item()

        test_loss = test_loss / len(dataloader)
        test_accuracy = test_correct.item() / len(dataloader.dataset)
        test_entropy = test_entropy / len(dataloader.dataset)
        test_max_p = test_max_p / len(dataloader.dataset)

        return test_loss, test_accuracy, test_entropy, test_max_p

    def sink_csv_figures(self):
        # logging misc ######################################
        accuracy_dynamics_df = pd.DataFrame(self.accuracy_dynamics)
        loss_dynamics_df = pd.DataFrame(self.loss_dynamics)
        entropy_dynamics_df = pd.DataFrame(self.entropy_dynamics)
        max_p_dynamics_df = pd.DataFrame(self.max_p_dynamics)

        metadata_dynamics_df = pd.DataFrame(self.metadata_dynamics)
        tosink_dataframes = {
            "accuracy": accuracy_dynamics_df,
            "loss": loss_dynamics_df,
            "normalised_entropy": entropy_dynamics_df,
            "max_p": max_p_dynamics_df,
            #
            "metadata": metadata_dynamics_df,
        }
        file_name = "_".join(tosink_dataframes.keys())
        ########################
        xlsx_writer = pd.ExcelWriter(
            "{}/{}.xlsx".format(self.summarydir, file_name), engine="xlsxwriter"
        )
        for dfname, dfdata in tosink_dataframes.items():
            dfdata.to_excel(xlsx_writer, sheet_name=dfname)
        xlsx_writer.close()

        ########################
        html_writer = open("{}/{}.html".format(self.summarydir, file_name), "w")
        for dfname, dfdata in tosink_dataframes.items():
            fig = ff.create_table(
                dfdata,
                index=True,
                colorscale=colorscale,
            )
            # Make text size larger
            for i in range(len(fig.layout.annotations)):
                fig.layout.annotations[i].font.size = 12

            html_writer.write(
                "<p style="
                + "text-align:center"
                + ">{}</p>".format(dfname)
                + fig.to_html()
                + "<p>&nbsp;&nbsp;</p>"
            )
        html_writer.close()

        ########################
        # save figures
        for dfname, dfdata in tosink_dataframes.items():
            dfdata.index = dfdata["epoch"]
            dfdata = dfdata.drop(columns=["epoch"])
            dfdata.plot.line()
            plt.savefig(
                "{}/{}.pdf".format(self.summarydir, dfname),
                dpi=100,
            )
