# CONFIGS = {
#     'ViT-B_16': configs.get_b16_config(),
#     'ViT-B_32': configs.get_b32_config(),
#     'ViT-L_16': configs.get_l16_config(),
#     'ViT-L_32': configs.get_l32_config(),
#     'ViT-H_14': configs.get_h14_config(),
#     'R50-ViT-B_16': configs.get_r50_b16_config(),
#     'testing': configs.get_testing(),
# }
import numpy as np

from proselflc.slices.networks.vit_modelconfig.modeling import (
    CONFIGS,
    VisionTransformer,
)


def ViT_B_16(params: dict = {"num_classes": 100}):
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(
        config,
        params["img_size"],
        zero_head=True,
        num_classes=params["num_classes"],
    )
    model.load_from(np.load(params["pretrained_dir"]))

    return model
