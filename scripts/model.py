import torch
import torch.nn as nn


CONFIGS = {
    "8→16→32": [8, 16, 32],
    "8→16":    [8, 16],
    "4→8":     [4, 8],
    "12→12":   [12, 12],
    "16→8":    [16, 8],
    "8→8":     [8, 8],
}


class SnoreCNN(nn.Module):
    def __init__(self, channels=None):
        super().__init__()
        if channels is None:
            channels = CONFIGS["12→12"]

        layers = []
        in_ch = 1
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(channels[-1], 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)


if __name__ == "__main__":
    x = torch.randn(1, 1, 64, 188)
    for name, channels in CONFIGS.items():
        model = SnoreCNN(channels)
        total = sum(p.numel() for p in model.parameters())
        out = model(x)
        print(f"{name:10s} | params: {total:>6,} | output: {out.shape}")
