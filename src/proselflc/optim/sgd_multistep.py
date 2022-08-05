import math

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from proselflc.exceptions import ParamException


# https://github.com/jeonsworld/ViT-pytorch/blob/main/utils/scheduler.py
# https://github.com/jeonsworld/ViT-pytorch/blob/main/train.py
class WarmupLinearSchedule(LambdaLR):
    """Linear warmup and then linear decay.
    Linearly increases learning rate from 0 to 1
    over `warmup_steps` training steps.
    Linearly decreases learning rate from 1. to 0.
    over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0.0,
            float(self.t_total - step)
            / float(max(1.0, self.t_total - self.warmup_steps)),
        )


class WarmupConstantSchedule(LambdaLR):
    """Linear warmup and then constant.
    Linearly increases learning rate schedule from 0 to 1
    over `warmup_steps` training steps.
    Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.0


class WarmupMultiStepSchedule(LambdaLR):
    """Linear warmup and then multi-step lr decay.
    Linearly increases learning rate from 0 to 1
    over `warmup_steps` training steps.
    Multi-step decreases learning rate.
    over remaining `t_total - warmup_steps` steps.
    """

    def __init__(
        self, optimizer, warmup_steps, milestones, gamma, t_total, last_epoch=-1
    ):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.milestones = milestones
        self.gamma = float(gamma)
        # TODO: add sanity check for milestones w.r.t. warmup_steps, t_total
        super(WarmupMultiStepSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        m_count = 0
        for milestone in self.milestones:
            if milestone <= step <= self.t_total:
                m_count = m_count + 1
        return max(0.0, math.pow(self.gamma, m_count))


class SGDMultiStep:
    """
    Setup SGD optimise with MultiStep learning rate scheduler

    Input:
        1. net.parameters()
        2. params, a dictory of key:value map.
            params["lr"]: float, e.g., 0.1
            params["milestones"]: e.g., [60, 120, 160]
            params["gamma"]: float, e.g., 0.2

    Remark:
        In this version, fix some parmas to make it simpler to use:
            params["momentum"] = 0.9
            params["weight_decay"] = 5e-4

    Return:
        1. optimiser
        2. train_scheduler

    TODO:
        unitests
    """

    def __init__(self, net_params, params):
        """
        Input:
            1. net_params: e.g., net.parameters()
            2. params, a dictory of key:value map.
                params["lr"]: float, e.g., 0.1
                params["milestones"]: e.g., [60, 120, 160]
                params["gamma"]: float, e.g., 0.2
        """
        # TODO: more sanity check

        if "lr" not in params.keys() or not isinstance(params["lr"], float):
            error_msg = (
                "The input params have no key of lr. "
                + "params["
                + "lr"
                + "] "
                + " has to be provided as a float data type."
            )
            raise (ParamException(error_msg))

        # if "milestones" not in params.keys() or not isinstance(
        #     params["milestones"], list
        # ):
        #     error_msg = (
        #         "The input params have no key of milestones. "
        #         + "params["
        #         + "milestones"
        #         + "] "
        #         + " has to be provided as a list of integers."
        #         + "E.g., params["
        #         + "milestones"
        #         + "] = [60, 120, 160]"
        #     )
        #     raise (ParamException(error_msg))
        #
        # if "gamma" not in params.keys() or not isinstance(params["gamma"], float):
        #     error_msg = (
        #         "The input params have no key of gamma. "
        #         + "params["
        #         + "gamma"
        #         + "] "
        #         + " has to be provided as a float data type."
        #     )
        #     raise (ParamException(error_msg))

        if (
            "layer_split_lr_scale" in params.keys()
            and params["layer_split_lr_scale"] is not None
        ):
            layer_split = params["layer_split_lr_scale"][0]
            lr_scale = params["layer_split_lr_scale"][1]

            net_params_list = list(net_params)
            backbone_params = net_params_list[:layer_split]
            fclayers_params = net_params_list[layer_split:]
            # backbone_params.append(fclayers_params[1])

            self.optimizer = optim.SGD(
                [
                    {
                        # backbone + bias
                        "params": backbone_params,
                        "lr": params["lr"],
                    },
                    {
                        # only for FC weight
                        "params": fclayers_params,
                        "lr": params["lr"] * lr_scale,
                    },
                ],
                momentum=params["momentum"],
                weight_decay=params["weight_decay"],
            )
        else:
            self.optimizer = optim.SGD(
                net_params,
                lr=params["lr"],
                momentum=params["momentum"],
                weight_decay=params["weight_decay"],
            )

        if "lr_scheduler" in params.keys():
            if params["lr_scheduler"] == "WarmupLinearSchedule":
                # 20220203: prior results, not stable at all.
                # therefore, I am working on decreasing lr
                self.lr_scheduler = WarmupLinearSchedule(
                    self.optimizer,
                    warmup_steps=params["warmup_iterations"]
                    // params["batch_accumu_steps"],
                    t_total=params["total_iterations"] // params["batch_accumu_steps"],
                )
            elif params["lr_scheduler"] == "WarmupConstantSchedule":
                self.lr_scheduler = WarmupConstantSchedule(
                    self.optimizer,
                    warmup_steps=params["warmup_iterations"]
                    // params["batch_accumu_steps"],
                )
            else:
                import numpy

                self.lr_scheduler = WarmupMultiStepSchedule(
                    self.optimizer,
                    warmup_steps=params["warmup_iterations"]
                    // params["batch_accumu_steps"],
                    t_total=params["total_iterations"] // params["batch_accumu_steps"],
                    milestones=list(
                        numpy.array(params["milestones"])
                        // params["batch_accumu_steps"]
                    ),
                    gamma=params["gamma"],
                )
        else:
            error_msg = (
                "The input params have no key of lr_scheduler. "
                + "params["
                + "lr_scheduler"
                + "] "
                + " has to be provided as one out of {}.".format(
                    [
                        "WarmupLinearSchedule",
                        "WarmupConstantSchedule",
                        "WarmupMultiStepSchedule",
                    ]
                )
            )
            raise ParamException(error_msg)
