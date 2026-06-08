import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, residual: bool):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.residual = residual and in_features == out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.norm(self.linear(x)))
        if self.residual:
            y = x + y
        return y


class ResidualMLP32(nn.Module):
    """
    32-hidden-layer residual MLP for CIFAR-10, matching the Newton-Muon paper.

    Hidden layer 0 maps 3072 -> width without a skip connection. The remaining
    hidden layers are width -> width residual blocks. The classifier is excluded
    from matrix-optimizer parameter names, following the paper's CIFAR setup.
    """

    def __init__(self, width: int = 512, depth: int = 32, num_classes: int = 10):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.input = ResidualMLPBlock(3072, width, residual=False)
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(width, width, residual=True)
            for _ in range(depth - 1)
        )
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)

    def matrix_param_names(self) -> set[str]:
        names = {"input.linear.weight"}
        names.update(f"blocks.{i}.linear.weight" for i in range(len(self.blocks)))
        return names
