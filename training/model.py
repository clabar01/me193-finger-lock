"""The finger-count CNN: three conv layers, each with ReLU and max pooling, then one linear layer.

Input: (N, C, 128, 128) from common.preprocess.preprocess: C = 3, or 1 for grayscale. Output: (N, 6) logits
for 0-5 fingers. The kernels shrink (7, 5, 3) as the channels grow (16, 32, 64).
About 132,000 parameters, roughly 0.5 MB as float32.
"""
from torch import nn

CLASSES = 6


class FingerCNN(nn.Module):
    def __init__(self, classes=CLASSES, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool2d(2),   # -> 16 x 64 x 64
            nn.Conv2d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),  # -> 32 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> 64 x 16 x 16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),  # active only while training; helps with this small dataset
            nn.Linear(64 * 16 * 16, classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def stage_outputs(self, x):
        """Feature maps after each conv+ReLU+pool block, then the logits."""
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i % 3 == 2:
                outputs.append(x)
        return outputs, self.classifier(x)
