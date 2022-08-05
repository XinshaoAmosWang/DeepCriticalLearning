import torch
from torch import Tensor

from proselflc.exceptions import ParamException

from .crossentropy import CrossEntropy


class ProSelfLC(CrossEntropy):
    """
    The implementation for progressive self label correction (CVPR 2021 paper).
    The target probability will be corrected by
    a predicted distributions, i.e., self knowledge.
        1. ProSelfLC is partially inspired by prior related work,
            e.g., Pesudo-labelling.
        2. ProSelfLC is partially theorectically bounded by
            early stopping regularisation.

    Inputs: two tensors for predictions and target.
        1. predicted probability distributions of shape (N, C)
        2. target probability  distributions of shape (N, C)
        3. current time (epoch/iteration counter).
        4. total time (total epochs/iterations)
        5. exp_base: the exponential base for adjusting epsilon
        6. counter: iteration or epoch counter versus total time.

    Outputs: scalar tensor, normalised by the number of examples.
    """

    def __init__(
        self,
        params: dict = None,
    ) -> None:
        super().__init__()
        self.total_epochs = params["total_epochs"]
        self.exp_base = params["exp_base"]
        self.counter = params["counter"]
        self.epsilon = None
        self.global_trust = None
        self.example_trust = None

        self.transit_time_ratio = params["transit_time_ratio"]
        # add support for batch accumulation
        self.batch_accumu_steps = params["batch_accumu_steps"]
        self.loss_mode_list = {
            "relative entropy": "with grad to targets",
            "cross entropy": "no grad to targets",
        }
        self.loss_mode = params["loss_mode"]
        if self.loss_mode not in self.loss_mode_list.keys():
            error_msg = (
                "self.loss_mode = "
                + str(self.loss_mode)
                + ", "
                + "not in [{}]".format(self.loss_mode_list.keys())
            )
            raise (ParamException(error_msg))

        # trust_mode_list
        self.trust_mode_list = {
            "global only": "global only",
            "global*(1-H(p)/H(u))": "global*(1-H(p)/H(u))",
            "global*max_p": "global*max_p",
        }
        self.trust_mode = params["trust_mode"]
        if self.trust_mode not in self.trust_mode_list.keys():
            error_msg = (
                "self.trust_mode = "
                + str(self.trust_mode)
                + ", "
                + "not in [{}]".format(self.trust_mode_list.keys())
            )
            raise (ParamException(error_msg))

        if not (self.exp_base >= 0):
            error_msg = (
                "self.exp_base = "
                + str(self.exp_base)
                + ". "
                + "The exp_base has to be no less than zero. "
            )
            raise (ParamException(error_msg))

        if not (isinstance(self.total_epochs, int) and self.total_epochs > 0):
            error_msg = (
                "self.total_epochs = "
                + str(self.total_epochs)
                + ". "
                + "The total_epochs has to be a positive integer. "
            )
            raise (ParamException(error_msg))

        if self.counter not in ["iteration", "epoch"]:
            error_msg = (
                "self.counter = "
                + str(self.counter)
                + ". "
                + "The counter has to be iteration or epoch. "
                + "The training time is counted by eithor of them. "
            )
            raise (ParamException(error_msg))

        if "total_iterations" in params.keys():
            # only exist when counter == "iteration"
            # add support for batch accumulation
            self.total_iterations = (
                params["total_iterations"] // self.batch_accumu_steps
            )

    def update_epsilon_progressive_adaptive(self, pred_probs, cur_time):
        with torch.no_grad():
            # global trust/knowledge
            if self.counter == "epoch":
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_epochs - self.transit_time_ratio
                )
            else:
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_iterations - self.transit_time_ratio
                )

            global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))

            # trust mode
            # the trade-off
            if self.trust_mode == "global only":
                # global only
                example_trust = torch.tensor(1.0)
                self.epsilon = global_trust * example_trust
            elif self.trust_mode == "global*(1-H(p)/H(u))":
                # global*(1-H(p)/H(u))
                # example-level trust/knowledge
                class_num = pred_probs.shape[1]
                H_pred_probs = torch.sum(-pred_probs * torch.log(pred_probs + 1e-8), 1)
                H_uniform = -torch.log(torch.tensor(1.0 / class_num))
                example_trust = 1 - H_pred_probs / H_uniform

                self.epsilon = global_trust * example_trust
                # from shape [N] to shape [N, 1]
                self.epsilon = self.epsilon[:, None]
                self.epsilon.requires_grad = False
            else:
                # global*max_p
                example_trust, _indexes = torch.max(pred_probs, 1)
                self.epsilon = global_trust * example_trust
                # from shape [N] to shape [N, 1]
                self.epsilon = self.epsilon[:, None]
                self.epsilon.requires_grad = False

            self.global_trust = global_trust
            self.example_trust = example_trust

    def forward(
        self,
        pred_probs: Tensor,
        target_probs: Tensor,
        cur_time: int,
        batch_mean_epsilon=None,
        calibrated_pred_probs=None,
    ) -> Tensor:
        """
        Inputs:
            1. predicted probability distributions of shape (N, C)
            2. target probability  distributions of shape (N, C)
            3. current time (epoch/iteration counter).

        Outputs:
            Loss: a scalar tensor, normalised by N.
        """
        if self.counter == "epoch":
            # cur_time indicate epoch
            if not (cur_time <= self.total_epochs and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_epochs)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))
        else:  # self.counter == "iteration":
            # cur_time indicate iteration
            # the value of cur_time will not change outside this function
            cur_time = cur_time // self.batch_accumu_steps
            if not (cur_time <= self.total_iterations and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_iterations)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))

        if calibrated_pred_probs is None:
            self.update_epsilon_progressive_adaptive(pred_probs.detach(), cur_time)
            new_target_probs = (
                1 - self.epsilon
            ) * target_probs + self.epsilon * pred_probs
        else:
            self.update_epsilon_progressive_adaptive(calibrated_pred_probs, cur_time)
            new_target_probs = (
                1 - self.epsilon
            ) * target_probs + self.epsilon * calibrated_pred_probs

        # filter
        # according to
        # https://discuss.pytorch.org/t/
        # filter-out-undesired-rows/28933/2
        # no much to learn if over 0.9 ???
        # 2022-02-07: I used 0.8 -> 40%s valid at the end
        # 2022-02-08: I use 0.90
        # _threshold = torch.tensor( 0.90 )
        # conditions = torch.squeeze(self.epsilon) < _threshold
        # pred_probs = pred_probs[conditions, :]
        # target_probs = target_probs[conditions, :]
        # self.epsilon = self.epsilon[conditions, :]

        # ########################
        if isinstance(batch_mean_epsilon, list) and len(batch_mean_epsilon) == 4:
            # this is a reference variable by assignment
            batch_mean_epsilon[0] = torch.mean(self.epsilon).item()
            batch_mean_epsilon[1] = pred_probs.shape[0]
            batch_mean_epsilon[2] = self.global_trust.item()
            batch_mean_epsilon[3] = torch.mean(self.example_trust).item()
        # ########################

        # assert new_target_probs.requires_grad == True
        # assert pred_probs.requires_grad == True
        # assert self.epsilon.requires_grad == False
        # assert target_probs.requires_grad == False
        # reuse CrossEntropy's forward computation
        if self.loss_mode == "relative entropy":
            # will have no effect if
            # using calibrated_pred_probs
            return super().forward(pred_probs, new_target_probs)
        else:  # "cross entropy": "no grad to targets"
            return super().forward(pred_probs, new_target_probs.detach())
