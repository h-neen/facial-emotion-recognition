"""
utils/config.py
All EA-Net hyperparameters in one dataclass.
Edit this file to tune the model — do not scatter magic numbers.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────────────────────
    project_root:   Path = Path(__file__).resolve().parent.parent
    fer_raw_root:   Path = Path("data/FER2013")
    fer_sr_root:    Path = Path("data/FER2013_SR")      # after super-resolution
    kdef_raw_root:  Path = Path("data/KDEF")
    kdef_aug_root:  Path = Path("data/KDEF_AUG")        # after augmentation
    checkpoint_dir: Path = Path("checkpoints")
    log_dir:        Path = Path("logs")
    results_dir:    Path = Path("results")
    lapsrn_model:   Path = Path("preprocessing/LapSRN_x4.pb")

    # ── Dataset ──────────────────────────────────────────────────────────────
    num_classes: int  = 7
    class_names: Tuple[str, ...] = (
        "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
    )
    image_size:  int  = 192          # after 4x super-resolution
    sr_scale:    int  = 4
    kdef_split:  Tuple[float, float, float] = (0.70, 0.15, 0.15)   # train/val/test
    random_seed: int  = 42

    # ── Model ────────────────────────────────────────────────────────────────
    backbone_out_channels: int = 512   # projection dim after each backbone
    cam_reduction:         int = 16    # channel attention reduction ratio
    dropout_rate:         float = 0.2

    # ── Training — Stage 1 (frozen backbone) ────────────────────────────────
    stage1_epochs:     int   = 20
    stage1_lr:        float  = 1e-3
    stage1_momentum:  float  = 0.9
    stage1_weight_decay: float = 1e-4

    # ── Training — Stage 2 (full fine-tune) ──────────────────────────────────
    stage2_epochs:     int   = 30
    stage2_lr:        float  = 5e-4
    stage2_momentum:  float  = 0.9
    stage2_weight_decay: float = 1e-4

    # ── Training — Shared ────────────────────────────────────────────────────
    batch_size:      int   = 48       # RTX 4050 6GB safe; reduce to 16 if OOM
    num_workers:     int   = 0        # Windows: keep at 0 to avoid spawn errors
    pin_memory:      bool  = True
    use_amp:         bool  = True     # FP16 mixed precision
    grad_clip:       float = 1.0      # max_norm for gradient clipping
    early_stop_patience: int = 20
    save_every_n_epochs: int = 5

    # ── LR Schedule (StepLR) ────────────────────────────────────────────────
    lr_step_size: int   = 15
    lr_gamma:    float  = 0.1

    def __post_init__(self):
        # Convert string paths to Path objects if needed
        for attr in ('project_root', 'fer_raw_root', 'fer_sr_root',
                     'kdef_raw_root', 'kdef_aug_root', 'checkpoint_dir',
                     'log_dir', 'results_dir', 'lapsrn_model'):
            val = getattr(self, attr)
            if not isinstance(val, Path):
                setattr(self, attr, Path(val))

    def make_dirs(self):
        """Create all output directories."""
        for d in (self.checkpoint_dir, self.log_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)


# Singleton — import this anywhere
CFG = Config()
