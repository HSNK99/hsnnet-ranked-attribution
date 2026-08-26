from dataclasses import dataclass
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# LOCKED ACC+ HSNNET MODEL DEFINITION
# Exact architecture used by the algorithm-specific binary detector bank.
# =============================================================================

PAYLOAD = "0.4bpp"
ROOT_DIR = r"${BOSSBASE_ROOT}"
COVER_DIR = os.path.join(ROOT_DIR, "cover")
STEGO_ROOT = os.path.join(ROOT_DIR, "stego")
OUT_ROOT = r".\outputs_acc_plus_all_algorithms_04bpp"

ALGO_FOLDER_MAP = {
    "WOW": "WOW",
    "S-UNIWARD": "S-UNIWARD",
    "HUGO": "HUGO",
    "MiPOD": "MiPOD",
    "HILL": "HILL",
}

EXPECTED_TOTAL_PARAMETERS = 7_293_009
EXPECTED_TRAINABLE_PARAMETERS = 7_292_609
EXPECTED_FROZEN_FRONTEND_PARAMETERS = 400


@dataclass
class CFG:
    algorithm: str = "WOW"
    payload: str = PAYLOAD
    root_dir: str = ROOT_DIR
    cover_dir: str = COVER_DIR
    stego_root: str = STEGO_ROOT
    stego_dir: str = ""
    out_root: str = OUT_ROOT
    out_dir: str = ""

    image_size: int = 256
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True

    epochs: int = 150
    lr: float = 8e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.01

    patience: int = 20
    early_stop_start_epoch: int = 70
    grad_clip: float = 1.0
    ema_decay: float = 0.999

    seed: int = 42
    use_amp: bool = True
    use_resize: bool = False
    save_plots: bool = True

    tlu_threshold: float = 3.0
    sigma_window: int = 3
    normalize_kernels: bool = True
    frontend_trainable: bool = False

    use_augmentation: bool = True
    d4_repeat_factor: int = 2
    label_smoothing: float = 0.02

    use_final_tta: bool = True
    tta_d4_count: int = 8


def make_cfg_for_algorithm(algo: str) -> CFG:
    if algo not in ALGO_FOLDER_MAP:
        raise ValueError(f"Unknown algorithm: {algo}")
    folder = ALGO_FOLDER_MAP[algo]
    stego_dir = os.path.join(STEGO_ROOT, folder, PAYLOAD, "stego")
    safe_algo = algo.replace("/", "_").replace("\\", "_")
    out_dir = os.path.join(OUT_ROOT, f"outputs_template_{safe_algo}_acc_plus")
    return CFG(
        algorithm=algo,
        payload=PAYLOAD,
        stego_dir=stego_dir,
        out_dir=out_dir,
    )


def _normalize_kernel_l1(k: torch.Tensor) -> torch.Tensor:
    s = k.abs().sum()
    return k / s if s > 0 else k


def _place_line_kernel(values, direction="h", mode="short") -> torch.Tensor:
    k = torch.zeros((5, 5), dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)

    if mode == "short":
        idx = [1, 2, 3]
    elif mode == "long":
        idx = [0, 2, 4]
    elif mode == "full":
        idx = [0, 1, 2, 3, 4]
    else:
        raise ValueError("mode must be short, long, or full")

    if len(values) != len(idx):
        raise ValueError("Kernel coefficient count does not match placement mode")

    if direction == "h":
        for c, v in zip(idx, values):
            k[2, c] = v
    elif direction == "v":
        for r, v in zip(idx, values):
            k[r, 2] = v
    elif direction == "d":
        for i, v in zip(idx, values):
            k[i, i] = v
    elif direction == "ad":
        for i, v in zip(idx, values):
            k[i, 4 - i] = v
    else:
        raise ValueError("direction must be h, v, d, or ad")
    return k


def get_statistical_prediction_filters(normalize: bool = True) -> torch.Tensor:
    kernels = []

    def add(k):
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, dtype=torch.float32)
        else:
            k = k.clone().float()
        kernels.append(_normalize_kernel_l1(k) if normalize else k)

    add(_place_line_kernel([-0.5, 1.0, -0.5], "h", "short"))
    add(_place_line_kernel([-0.5, 1.0, -0.5], "v", "short"))
    add(_place_line_kernel([-0.5, 1.0, -0.5], "d", "short"))
    add(_place_line_kernel([-0.5, 1.0, -0.5], "ad", "short"))

    add([
        [0, 0, 0, 0, 0],
        [0, 0, -0.25, 0, 0],
        [0, -0.25, 1.0, -0.25, 0],
        [0, 0, -0.25, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    add([
        [0, 0, 0, 0, 0],
        [0, -0.125, -0.125, -0.125, 0],
        [0, -0.125, 1.0, -0.125, 0],
        [0, -0.125, -0.125, -0.125, 0],
        [0, 0, 0, 0, 0],
    ])

    add(_place_line_kernel([-0.5, 1.0, -0.5], "h", "long"))
    add(_place_line_kernel([-0.5, 1.0, -0.5], "v", "long"))
    add(_place_line_kernel([-0.5, 1.0, -0.5], "d", "long"))
    add(_place_line_kernel([-0.5, 1.0, -0.5], "ad", "long"))

    add(_place_line_kernel(
        [-0.25, -0.25, 1.0, -0.25, -0.25], "h", "full"
    ))
    add(_place_line_kernel(
        [-0.25, -0.25, 1.0, -0.25, -0.25], "v", "full"
    ))

    add(_place_line_kernel(
        [0.25, -0.75, 1.0, -0.75, 0.25], "h", "full"
    ))
    add(_place_line_kernel(
        [0.25, -0.75, 1.0, -0.75, 0.25], "v", "full"
    ))
    add(_place_line_kernel(
        [0.25, -0.75, 1.0, -0.75, 0.25], "d", "full"
    ))
    add(_place_line_kernel(
        [0.25, -0.75, 1.0, -0.75, 0.25], "ad", "full"
    ))

    return torch.stack(kernels, dim=0).unsqueeze(1)


class StatisticalResidualFrontEnd(nn.Module):
    def __init__(
        self,
        in_channels=1,
        tlu_threshold=3.0,
        sigma_window=3,
        eps=1e-4,
        normalize_kernels=True,
        trainable=False,
    ):
        super().__init__()
        if sigma_window % 2 == 0:
            raise ValueError("sigma_window must be odd")

        kernels = get_statistical_prediction_filters(
            normalize=normalize_kernels
        )
        self.conv = nn.Conv2d(
            in_channels,
            kernels.shape[0],
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )

        if in_channels != 1:
            kernels = kernels.repeat(1, in_channels, 1, 1) / float(in_channels)

        with torch.no_grad():
            self.conv.weight.copy_(kernels)

        self.conv.weight.requires_grad = bool(trainable)
        self.tlu_threshold = float(tlu_threshold)
        self.sigma_window = int(sigma_window)
        self.eps = float(eps)

    def local_variance_normalize(self, residual):
        pad = self.sigma_window // 2
        mu = F.avg_pool2d(
            residual,
            self.sigma_window,
            stride=1,
            padding=pad,
        )
        mu2 = F.avg_pool2d(
            residual * residual,
            self.sigma_window,
            stride=1,
            padding=pad,
        )
        variance = torch.clamp(mu2 - mu * mu, min=0.0)
        std = torch.sqrt(variance + self.eps)
        return residual / (std + self.eps)

    def forward(self, x):
        residual = self.conv(x)
        normalized = self.local_variance_normalize(residual)
        return torch.clamp(
            normalized,
            -self.tlu_threshold,
            self.tlu_threshold,
        )


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                k,
                stride=s,
                padding=p,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        hidden = max(ch // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class ResidualBlock(nn.Module):
    def __init__(self, ch, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.se = SEBlock(ch) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) + x
        return self.relu(out)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.max = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        return torch.cat([self.avg(x), self.max(x)], dim=1)


class hsnnet(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.frontend = StatisticalResidualFrontEnd(
            in_channels=1,
            tlu_threshold=cfg.tlu_threshold,
            sigma_window=cfg.sigma_window,
            eps=1e-4,
            normalize_kernels=cfg.normalize_kernels,
            trainable=cfg.frontend_trainable,
        )

        self.stem = nn.Sequential(
            ConvBNReLU(16, 32),
            ResidualBlock(32),
            nn.AvgPool2d(2, 2),
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(32, 64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.AvgPool2d(2, 2),
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(64, 128),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.AvgPool2d(2, 2),
        )
        self.stage4 = nn.Sequential(
            ConvBNReLU(128, 256),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.AvgPool2d(2, 2),
        )
        self.stage5 = nn.Sequential(
            ConvBNReLU(256, 384),
            ResidualBlock(384),
        )

        self.pool = AdaptiveConcatPool2d()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(384 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool(x)
        return self.head(x)


HassanNetV2_SRM = hsnnet


def _apply_d4_batch(x, g: int):
    if g == 0:
        return x
    if g == 1:
        return torch.rot90(x, k=1, dims=[2, 3])
    if g == 2:
        return torch.rot90(x, k=2, dims=[2, 3])
    if g == 3:
        return torch.rot90(x, k=3, dims=[2, 3])
    if g == 4:
        return torch.flip(x, dims=[3])
    if g == 5:
        return torch.flip(x, dims=[2])
    if g == 6:
        return torch.rot90(
            torch.flip(x, dims=[3]),
            k=1,
            dims=[2, 3],
        )
    if g == 7:
        return torch.rot90(
            torch.flip(x, dims=[2]),
            k=1,
            dims=[2, 3],
        )
    raise ValueError(f"D4 element index out of range: {g}")


def architecture_counts(cfg: CFG | None = None):
    cfg = cfg or CFG()
    model = hsnnet(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frontend = sum(p.numel() for p in model.frontend.parameters())
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frontend_parameters": int(frontend),
    }
