# Synchronized archive/re-export of the active frequency fusion modules.
#
# The real training path is:
#   train.py -> models/yolo_test.py -> models/common.py -> HAFFormer
#
# Keep this file thin so it cannot drift away from the implementation actually
# used by training.

from models.common import (  # noqa: F401
    EOTChannelPrior,
    ExpMaskFrequencyCrossAttention,
    FourierRoutingLogPriorV2,
    FrequencyCrossAttention_S,
    FrequencyResponseGate,
    ForegroundSparseFusionMask,
    HAFFormer,
    Windowed2DGOATCrossAttention,
    init_complex_residual_bases,
    positive_temperature,
    resize_complex_weight,
)


if __name__ == '__main__':
    import torch

    block = HAFFormer(64)
    rgb = torch.randn(2, 64, 20, 20)
    ir = torch.randn(2, 64, 20, 20)
    out = block([rgb, ir])
    print(out.shape)
