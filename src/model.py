import torch.nn as nn
import torchvision

def make_model(version, dropout, hidden_size, hidden_depth):
  assert version in [1, 2, 3]

  if version == 1:
    return nn.Sequential(
      # --
      nn.Conv2d(3, 32, kernel_size=5),
      nn.BatchNorm2d(32),
      nn.ReLU(True),
      nn.MaxPool2d(2),
      # --
      nn.Conv2d(32, 64, kernel_size=3),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.MaxPool2d(2),
      # --
      nn.Conv2d(64, 128, kernel_size=3),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.MaxPool2d(2),
      # --
      nn.Conv2d(128, 256, kernel_size=3),
      nn.BatchNorm2d(256),
      nn.ReLU(True),
      nn.MaxPool2d(2),
      # --
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(),
      nn.Dropout(dropout),
      # --
      nn.Linear(256, 8)
    )

  if version == 2:
    base = torchvision.models.mobilenet.mobilenet_v3_small(pretrained=True, progress=True)
    for param in base.parameters():
        param.requires_grad = False

    lastconv_size = base.classifier[0].in_features
    base.classifier = nn.Sequential(
      nn.Linear(lastconv_size, 256),
      nn.ReLU(True),
      nn.Dropout(dropout),
      nn.Linear(256, 8)
    )
    return base

  if version == 3:
    features = torchvision.models.mobilenet_v3_large(pretrained=True, progress=True).features
    for param in features.parameters():
        param.requires_grad = False

    out_channels = features[-1].out_channels
    assert out_channels == 960, f"out_channels = {out_channels}"

    blocks = []
    for _ in range(hidden_depth):
      blocks.extend([
        nn.Conv2d(hidden_size, hidden_size, kernel_size=3),
        nn.ReLU(True)
      ])

    return nn.Sequential(
      features,
      nn.Conv2d(out_channels, hidden_size, kernel_size=3),
      nn.ReLU(True),
      *blocks,
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(),
      nn.Dropout(dropout),
      nn.Linear(hidden_size, 8)
    )
