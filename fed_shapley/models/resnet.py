"""
models/resnet.py — Dataset-adapted ResNet-18 factory.

Modifications per dataset:
- CIFAR-10 / CIFAR-100 (32×32):
    Replace 7×7 conv (stride=2) with 3×3 conv (stride=1) and remove first MaxPool.
    This preserves spatial resolution for small images.
- Tiny ImageNet (64×64):
    Replace first MaxPool with identity to reduce over-downsampling.
    Keep 7×7 conv but set stride=1 to match spatial dimensions better.

All variants use the standard torchvision ResNet-18 backbone.
"""

import torch.nn as nn
import torchvision.models as tv_models


def get_model(
    model_name: str,
    num_classes: int,
    dataset: str,
) -> nn.Module:
    """Construct a dataset-adapted ResNet-18 model.

    Args:
        model_name: Model identifier; currently only 'resnet18' is supported.
        num_classes: Number of output classes (10, 100, or 200).
        dataset: Dataset name used to select architecture adaptations.
            One of: 'cifar10', 'cifar100', 'tinyimagenet'.

    Returns:
        nn.Module: Adapted ResNet-18 ready for training.

    Raises:
        ValueError: If model_name is not 'resnet18'.

    Notes:
        - For CIFAR, the first conv is replaced with kernel_size=3, stride=1,
          padding=1, and the MaxPool layer is replaced with an identity.
          This follows the common practice for 32×32 inputs (e.g., He et al.,
          SimCLR-CIFAR, BYOL-CIFAR configurations).
        - For TinyImageNet, the MaxPool is removed (identity) to avoid excessive
          downsampling from 64×64 to 4×4 before layer1.
        - weights=None uses random initialization (no pretrained weights).
    """
    if model_name != "resnet18":
        raise ValueError(f"Unsupported model: {model_name}. Only 'resnet18' is supported.")

    # Build base ResNet-18 without pretrained weights
    model = tv_models.resnet18(weights=None, num_classes=num_classes)

    if dataset in ("cifar10", "cifar100"):
        # -- CIFAR adaptation --
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace with: Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        # Remove MaxPool (would reduce 32→16 before ResBlocks)
        model.maxpool = nn.Identity()

    elif dataset == "tinyimagenet":
        # -- Tiny ImageNet adaptation --
        # Replace MaxPool with identity to keep 64×64 → 32×32 instead of 64→8
        # The 7×7 conv with stride=2 already reduces 64→32; MaxPool would then
        # give 16×16 which is too small for 4 residual stages.
        model.maxpool = nn.Identity()

    return model
