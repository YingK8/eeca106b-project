import torch
from torch import nn
from torchvision import models


class KeypointCNN(nn.Module):
    """Default perseus keypoint CNN model trained by fine-tuning resnet."""

    def __init__(self, n_keypoints: int = 8, num_channels: int = 3, H: int = 256, W: int = 256) -> None:
        """Initialize the keypoint CNN model.

        Args:
            n_keypoints: The number of keypoints to predict.
            num_channels: The number of channels in the input images.
            H: The height of the input images.
            W: The width of the input images.
        """
        super(KeypointCNN, self).__init__()
        # Load a prebuilt ResNet (e.g., ResNet18) and modify it
        self.resnet = models.resnet18(weights="DEFAULT")
        self.n_keypoints = n_keypoints
        self.num_channels = num_channels
        self.H = H
        self.W = W

        # Adjust the first convolutional layer if the input has a different number of channels than 3
        if num_channels != 3:  # noqa: PLR2004
            self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the average pooling and the final fully connected layer
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2 * n_keypoints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x: tensor of input images, shape=(batch_size, num_channels, H, W).
        """
        return self.resnet(x)
