"""ResNet-18 adapted with a CIFAR-10 variant for 32x32 CIFAR images."""

from torch import nn
from torchvision.models import resnet18

NUM_CLASSES = 10


def resnet18_cifar(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Randomly initialises a ResNet-18 model for CIFAR sized images.

    The CIFAR variant uses a 3x3 stride-1 first convolution and no max-pooling. This
    keeps the full 32x32 resolution.

    Args:
        num_classes (int): Number of classes in the output.

    Returns:
        model (nn.Module): Randomly initialised ResNet-18 model.
    """
    model: nn.Module = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
