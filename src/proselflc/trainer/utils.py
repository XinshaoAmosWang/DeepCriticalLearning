import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from proselflc.exceptions import ParamException


def logits2probs_softmax(logits):
    """
    Transform logits to probabilities using exp function and normalisation

    Input:
        logits with shape: (N, C)
        N means the batch size or the number of instances.
        C means the number of training classes.

    Output:
        probability vectors of shape (N, C)
    """
    # reimplementation of F.softmax(logits)
    # or: torch.nn.Softmax(dim=1)(logits)
    # per instance:
    # subtract max logit for numerical issues
    # subtractmax_logits = logits - torch.max(logits, dim=1, keepdim=True).values
    # exp_logits = torch.exp(subtractmax_logits)
    # sum_logits = torch.sum(exp_logits, dim=1, keepdim=True)
    # return exp_logits / sum_logits
    return torch.nn.Softmax(dim=1)(logits)


@torch.no_grad()
def intlabels2onehotmatrix(device: str, class_num, intlabels) -> Tensor:
    target_probs = np.zeros((len(intlabels), class_num), dtype=np.float32)
    for i in range(len(intlabels)):
        # default ignore index
        if intlabels[i] == -100:
            pass
        else:
            target_probs[i][intlabels[i]] = 1
    target_probs = torch.tensor(target_probs)
    if device == "gpu":
        target_probs = target_probs.cuda()
    return target_probs


def intlabel2onehot(intlabel, class_num) -> np.ndarray:
    """
    intlabel in the class index: [0, class_num-1]
    """
    if intlabel not in list(range(class_num)):
        error_msg = "intlabe: {}".format(
            intlabel
        ) + " not in the range of [0, {}]".format(class_num - 1)
        raise ParamException(error_msg)

    target_probs = np.zeros(class_num, dtype=np.float32)
    target_probs[intlabel] = 1
    return target_probs


def save_figures(fig_save_path="", y_inputs=[], fig_legends=[], xlabel="", ylabel=""):
    colors = ["r", "b"]
    linestyles = ["solid", "dashdot"]
    x = torch.arange(len(y_inputs[0]))
    #
    fig, ax = plt.subplots()
    for y_input, color, linestyle, fig_legend in zip(
        y_inputs, colors, linestyles, fig_legends
    ):
        ax.plot(x, y_input, color=color, linestyle=linestyle, label=fig_legend)
    # legend = ax.legend(loc="upper right")
    ax.legend(loc="upper right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #
    plt.savefig(fig_save_path, dpi=100)
