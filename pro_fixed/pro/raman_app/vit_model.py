"""
vit_model.py — a compact Vision Transformer adapted for 1D Raman spectra.

The classic ViT ("An Image is Worth 16x16 Words", Dosovitskiy et al. 2021)
splits a 2D image into patches; here each preprocessed spectrum (a 1D
sequence of intensities on a common wavenumber grid) is resampled to a
fixed length and split into contiguous 1D patches.  Patches are linearly
embedded, a CLS token is prepended, position embeddings are added, and the
sequence runs through standard transformer encoder blocks; the CLS token
heads the classifier.

Pure PyTorch — no torchvision/timm required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    torch = nn = Dataset = None          # class bodies below must still
    HAS_TORCH = False                    # IMPORT on a torch-less box


class _TorchRequired:
    """Placeholder base so the module imports without torch; touching
    the model classes then gives the actionable message instead of a
    NameError at import time (fixed 2026-09-05)."""

    def __init__(self, *args, **kwargs):
        require_torch()                  # raises SystemExit with instructions


_MODULE_BASE = nn.Module if HAS_TORCH else _TorchRequired
_DATASET_BASE = Dataset if HAS_TORCH else _TorchRequired


def require_torch():
    if not HAS_TORCH:
        raise SystemExit(
            "PyTorch is not installed.  Install it with:\n"
            "    pip install torch\n"
            "(add a CUDA build from download.pytorch.org to train on GPU)"
        )


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class ViTConfig:
    seq_len: int = 512        # fixed resampled spectrum length
    patch_size: int = 32      # 512 / 32 = 16 patch tokens
    dim: int = 128            # embedding dimension
    depth: int = 4            # transformer encoder blocks
    heads: int = 4            # attention heads (dim % heads == 0)
    mlp_ratio: float = 2.0    # MLP hidden dim = ratio * dim
    dropout: float = 0.1
    n_classes: int = 2        # set at build time

    @property
    def n_patches(self) -> int:
        return self.seq_len // self.patch_size

    def validate(self) -> "ViTConfig":
        c = ViTConfig(**asdict(self))
        c.seq_len = max(16, int(c.seq_len))
        c.patch_size = int(np.clip(c.patch_size, 4, c.seq_len))
        # make seq_len an exact multiple of patch_size
        c.seq_len = c.n_patches * c.patch_size
        c.dim = max(16, int(c.dim))
        c.heads = int(np.clip(c.heads, 1, c.dim))
        while c.dim % c.heads != 0:
            c.heads -= 1
        return c


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class PatchEmbed1D(_MODULE_BASE):
    """Linear projection of contiguous spectrum patches (via Conv1d)."""

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.proj = nn.Conv1d(1, cfg.dim, kernel_size=cfg.patch_size,
                              stride=cfg.patch_size)

    def forward(self, x):                  # x: (B, seq_len)
        return self.proj(x.unsqueeze(1))   # -> (B, dim, n_patches)


class EncoderBlock(_MODULE_BASE):
    """Pre-norm transformer encoder block (MHSA + MLP, both residual)."""

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.attn = nn.MultiheadAttention(cfg.dim, cfg.heads,
                                          dropout=cfg.dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(cfg.dim)
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.dim))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.drop(self.attn(h, h, h, need_weights=False)[0])
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class SpectralViT(_MODULE_BASE):
    """Vision Transformer over 1D Raman spectra (see module docstring)."""

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg.validate()
        c = self.cfg
        self.patch_embed = PatchEmbed1D(c)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, c.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, c.n_patches + 1, c.dim))
        self.blocks = nn.ModuleList(EncoderBlock(c) for _ in range(c.depth))
        self.norm = nn.LayerNorm(c.dim)
        self.head = nn.Linear(c.dim, c.n_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):                            # x: (B, seq_len)
        tokens = self.patch_embed(x).transpose(1, 2)  # (B, N, dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        z = torch.cat([cls, tokens], dim=1) + self.pos_embed
        for blk in self.blocks:
            z = blk(z)
        return self.head(self.norm(z[:, 0]))          # (B, n_classes)


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
class SpectraDataset(_DATASET_BASE):
    """(X, y) tensors of preprocessed, resampled spectra."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        self.y = torch.as_tensor(np.asarray(y, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def resample_to_len(X: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample each spectrum row to exactly `length` points."""
    X = np.asarray(X, dtype=float)
    if X.shape[1] == length:
        return X
    length = int(max(4, length))
    src = np.linspace(0.0, 1.0, X.shape[1])
    dst = np.linspace(0.0, 1.0, length)
    return np.vstack([np.interp(dst, src, row) for row in X])


# --------------------------------------------------------------------------
# Checkpoint I/O
# --------------------------------------------------------------------------
def save_checkpoint(path: str, model: SpectralViT, classes: list[str],
                    grid: np.ndarray, preprocess_params: dict,
                    splits: dict[str, list[str]], history: dict,
                    **extra):
    """
    Save everything vit_test.py needs to reproduce the evaluation:
    weights + config, class order, wavenumber grid, preprocessing params,
    the train/val/test filename split, the epoch history and any extras.
    """
    ckpt = {
        "format_version": 1,
        "model_config": asdict(model.cfg),
        "classes": list(classes),
        "grid": np.asarray(grid, dtype=float),
        "preprocess_params": dict(preprocess_params),
        "splits": {k: list(v) for k, v in splits.items()},
        "history": {k: [float(v) for v in vals]
                    for k, vals in history.items()},
        "state_dict": model.state_dict(),
    }
    ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str) -> tuple[SpectralViT, dict]:
    """Load a checkpoint -> (model in eval mode, full checkpoint dict)."""
    require_torch()
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only kwarg
        ckpt = torch.load(path, map_location="cpu")
    cfg = ViTConfig(**ckpt["model_config"]).validate()
    model = SpectralViT(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
