import torchvision
import torch
import torch.nn as nn


def resnet18(num_classes):
    # Build a fresh ResNet-18 backbone for classifier training/inference.
    resnet = torchvision.models.resnet18(weights=None)
    # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # CIFAR is converted from RGB to Grayscale
    num_channel = 3

    resnet.conv1 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # resnet.maxpool = nn.Identity()
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)

    return resnet


def load_resnet18_model_weights(model_path, num_classes, device="cpu"):
    """
    Build ResNet-18 with the expected classifier head and load weights from a checkpoint.

    Supports either a raw state_dict checkpoint or a wrapped checkpoint with
    a `state_dict` key.
    """
    model = resnet18(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model