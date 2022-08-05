import torch.nn as nn
import torchvision.models as models


def resnet50_tv(
    params: dict = {
        "num_classes": 14,
        "pretrained": True,
        "freeze_bn": True,
        "dropout": 0.2,
    }
):
    model = models.resnet50(pretrained=params["pretrained"])
    # overwrite fc layer
    if "dropout" in params.keys() and params["dropout"] > 0.0:
        # dropout probability of an element to be zeroed. Default: 0.5
        model.fc = nn.Sequential(
            nn.Dropout(p=params["dropout"]),
            nn.Linear(2048, params["num_classes"]),
        )
    else:
        model.fc = nn.Linear(2048, params["num_classes"])
    # freeze_bn?
    if "freeze_bn" in params.keys() and params["freeze_bn"]:
        for _layer in model.modules():
            if isinstance(_layer, nn.BatchNorm2d):
                # eval mode
                _layer.eval()
                _layer.weight.requires_grad = False
                _layer.bias.requires_grad = False
    return model
