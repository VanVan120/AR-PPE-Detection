"""A self-contained MS-TCN baseline for Assembly101 temporal action segmentation.

MS-TCN (Farha & Gall, CVPR 2019) is a clean, well-understood, fully-convolutional
temporal segmentation model: a stack of stages, each a set of dilated residual 1-D
convolutions, refining a per-frame class prediction. It runs comfortably on a laptop
GPU over the precomputed 2048-D TSM features.

Why MS-TCN here and not C2F-TCN: C2F-TCN's value is its *pretrained checkpoint* (the
exact val sanity target in the README), which only loads into the official module.
This MS-TCN is a self-contained, trainable baseline that proves the whole
train->eval->score pipeline end-to-end and can be trained on the real features
directly. Drop in the official C2F-TCN + checkpoint later for the canonical number;
the training/eval/metrics code around it is unchanged.

Input:  (B, dim, T) features. Output: (num_stages, B, num_classes, T) stage logits
(the last stage is the prediction; all stages are supervised during training).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3,
                                      padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class SingleStageTCN(nn.Module):
    def __init__(self, num_layers: int, num_f_maps: int, dim: int, num_classes: int):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        return self.conv_out(out)


class MSTCN(nn.Module):
    def __init__(self, num_stages: int = 4, num_layers: int = 10, num_f_maps: int = 64,
                 dim: int = 2048, num_classes: int = 202):
        super().__init__()
        self.num_classes = num_classes
        # Persisted with the weights so a checkpoint reconstructs its own architecture.
        self.arch = {"num_stages": num_stages, "num_layers": num_layers,
                     "num_f_maps": num_f_maps, "dim": dim, "num_classes": num_classes}
        self.stage1 = SingleStageTCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([
            copy.deepcopy(SingleStageTCN(num_layers, num_f_maps, num_classes, num_classes))
            for _ in range(num_stages - 1)])

    def forward(self, x):
        out = self.stage1(x)
        outputs = [out]
        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out)
        return torch.stack(outputs)     # (num_stages, B, num_classes, T)


def save_checkpoint(model: "MSTCN", path: str) -> None:
    """Save weights + architecture so the model can be reconstructed on load."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": dict(model.arch)}, path)


def load_model(path: str, device: str = "cpu") -> "MSTCN":
    """Reconstruct an MSTCN from a checkpoint saved by `save_checkpoint`, eval mode."""
    ckpt = torch.load(path, map_location=device)
    if not (isinstance(ckpt, dict) and "state_dict" in ckpt and "arch" in ckpt):
        raise ValueError(f"checkpoint '{path}' is missing 'arch'/'state_dict' "
                         "(train with the current train.py, which uses save_checkpoint)")
    model = MSTCN(**ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


class MSTCNLoss(nn.Module):
    """Standard MS-TCN objective: per-frame cross-entropy on every stage + a
    truncated temporal-smoothness (MSE on log-softmax step differences)."""

    def __init__(self, num_classes: int, smooth_weight: float = 0.15,
                 smooth_clamp: float = 16.0, ignore_index: int = -100):
        super().__init__()
        self.num_classes = num_classes
        self.smooth_weight = smooth_weight
        self.smooth_clamp = smooth_clamp
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, outputs, target):
        """outputs: (S, B, C, T); target: (B, T) int64."""
        loss = 0.0
        for p in outputs:                                   # p: (B, C, T)
            loss = loss + self.ce(p.transpose(2, 1).reshape(-1, self.num_classes),
                                  target.reshape(-1))
            smooth = self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                              F.log_softmax(p.detach()[:, :, :-1], dim=1))
            loss = loss + self.smooth_weight * torch.clamp(
                smooth, min=0, max=self.smooth_clamp).mean()
        return loss
