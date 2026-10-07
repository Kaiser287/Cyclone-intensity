"""
Intensity predictor cho model ResNet18 (1 kênh IR) train trên Cyclone Image Dataset
(TCIR, mọi basin). Checkpoint: models/intensity_best.pt
    dict {"model": state_dict (FP16), "stats": {...}, "in_ch": 1, "epoch": int}

THANG ĐO DỮ LIỆU TRAIN (quan trọng):
  - Ảnh IR dạng 0-255, TỐI = LẠNH (mây cao, bão mạnh), SÁNG = ẤM (bão yếu).
    Tâm bão mạnh trung bình ~49, bão yếu ~148 (tương quan với Vmax r = -0.64).
  - Ảnh IR vệ tinh thông thường thì ngược lại: TRẮNG = LẠNH -> phải đảo trước khi đưa vào model.

Đầu vào hợp lệ cho predict():
  - .npy / ndarray giá trị 0-255 (đúng định dạng dataset)       -> chính xác nhất
  - .npy / ndarray Kelvin (giá trị > 260)                       -> quy đổi xấp xỉ
  - PNG/JPG grayscale kiểu vệ tinh (trắng = lạnh)               -> đảo cực, xấp xỉ
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CROP = 128

# Quy đổi Kelvin -> thang 0-255 của dataset (lạnh -> 0, ấm -> 255). Xấp xỉ, không phải hiệu chuẩn thật.
BT_COLD = 180.0
BT_WARM = 310.0
KELVIN_THRESHOLD = 260.0  # mảng float có max > ngưỡng này được coi là Kelvin

REQUIRED_STATS = ("img_mean", "img_std", "vmax_mean", "vmax_std")

# Thang Saffir-Simpson (knots) – cùng cách chia khi đánh giá test
CATEGORIES = [
    (34,  "TD", "Tropical Depression", "#4CAF50"),
    (64,  "TS", "Tropical Storm",      "#CDDC39"),
    (83,  "C1", "Typhoon Cat 1",       "#FFC107"),
    (96,  "C2", "Typhoon Cat 2",       "#FF9800"),
    (113, "C3", "Typhoon Cat 3",       "#FF5722"),
    (137, "C4", "Typhoon Cat 4",       "#F44336"),
    (999, "C5", "Super Typhoon Cat 5", "#9C27B0"),
]


# ---------------------------------------------------------------- model
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _clean_keys(sd):
    out = {}
    for k, v in sd.items():
        for p in ("module.", "_orig_mod."):
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v.float() if torch.is_tensor(v) and v.is_floating_point() else v  # FP16 -> FP32
    for p in ("backbone.", "net.", "model.", "resnet."):
        if all(k.startswith(p) for k in out):
            out = {k[len(p):]: v for k, v in out.items()}
            break
    return out


def build_resnet18(sd):
    """Dựng ResNet18 khớp với state_dict (số kênh vào, kiểu head fc)."""
    m = models.resnet18(weights=None)
    in_ch = sd["conv1.weight"].shape[1]
    m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if "fc.weight" in sd:
        m.fc = nn.Linear(sd["fc.weight"].shape[1], sd["fc.weight"].shape[0])
    else:
        lin_idx = sorted({int(k.split(".")[1]) for k in sd
                          if k.startswith("fc.") and k.endswith(".weight")})
        layers = []
        for i in range(max(lin_idx) + 1):
            if i in lin_idx:
                w = sd[f"fc.{i}.weight"]
                layers.append(nn.Linear(w.shape[1], w.shape[0]))
            else:
                layers.append(nn.ReLU())  # Dropout = identity khi eval
        m.fc = nn.Sequential(*layers)
    return m


def _scalar(x):
    return float(np.ravel(np.asarray(x))[0])


# ---------------------------------------------------------------- predictor
class IntensityPredictor:
    def __init__(self, checkpoint_path, device=DEVICE):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.device = device

        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:  # torch cũ không có weights_only
            ckpt = torch.load(checkpoint_path, map_location="cpu")

        sd = _clean_keys(_extract_state_dict(ckpt))
        self.model = build_resnet18(sd)
        try:
            self.model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            raise RuntimeError(f"Kiến trúc không khớp checkpoint. 8 key đầu: {list(sd)[:8]}\n{e}")
        self.model.to(device).float().eval()

        # Stats BẮT BUỘC lấy từ checkpoint (stats Kelvin cũ không dùng được cho model này)
        src = ckpt.get("stats", {}) if isinstance(ckpt, dict) else {}
        missing = [k for k in REQUIRED_STATS if k not in src]
        if missing:
            raise KeyError(f"Checkpoint thiếu stats: {missing}. Có: {list(src)}")
        self.stats = {k: _scalar(src[k]) for k in REQUIRED_STATS}
        self.crop = int(src.get("crop", ckpt.get("crop", DEFAULT_CROP)))
        self.in_ch = int(ckpt.get("in_ch", 1))
        self.epoch = ckpt.get("epoch")

    # ---------- input -> mảng 0-255 theo thang dataset (tối = lạnh), shape (H, W)
    @staticmethod
    def kelvin_to_native(bt):
        g = (np.asarray(bt, dtype=np.float32) - BT_COLD) / (BT_WARM - BT_COLD)
        return np.clip(g, 0, 1) * 255.0

    @staticmethod
    def satellite_to_native(gray_uint8):
        """Ảnh IR vệ tinh thường (trắng = lạnh) -> thang dataset (tối = lạnh)."""
        return 255.0 - gray_uint8.astype(np.float32)

    def to_native(self, x, png_white_is_cold=True):
        """Trả về (array_0_255, is_approx)."""
        if isinstance(x, str):
            x = np.load(x) if x.lower().endswith(".npy") else Image.open(x)
        if isinstance(x, Image.Image):
            g = np.array(x.convert("L"))
            arr = self.satellite_to_native(g) if png_white_is_cold else g.astype(np.float32)
            return arr, True

        arr = np.asarray(x)
        if arr.ndim == 3:  # (H, W, C) hoặc (C, H, W) -> kênh IR là index 0
            arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
        arr = arr.astype(np.float32)
        if np.nanmax(arr) > KELVIN_THRESHOLD:
            return self.kelvin_to_native(arr), True
        return arr, False  # đã đúng thang 0-255 của dataset

    # giữ tên cũ để app.py hiện tại không phải sửa
    def to_bt(self, x):
        return self.to_native(x)

    def _prepare(self, img):
        img = np.nan_to_num(img, nan=self.stats["img_mean"])
        h, w = img.shape
        t = torch.from_numpy(img).float()[None, None]
        c = self.crop
        if h >= c and w >= c and abs(h - w) <= 2 and h <= 260:
            top, left = (h - c) // 2, (w - c) // 2  # ảnh TCIR gốc: center crop như lúc train
            t = t[..., top:top + c, left:left + c]
        else:
            t = F.interpolate(t, size=(c, c), mode="bilinear", align_corners=False)
        t = (t - self.stats["img_mean"]) / self.stats["img_std"]
        return t.to(self.device)

    # ---------- inference
    @torch.no_grad()
    def predict(self, x, tta=True, png_white_is_cold=True):
        img, approx = self.to_native(x, png_white_is_cold)
        t = self._prepare(img)
        batch = torch.cat([torch.rot90(t, k, dims=(2, 3)) for k in range(4)]) if tta else t
        out = self.model(batch).float().view(-1).mean().item()
        vmax = float(max(out * self.stats["vmax_std"] + self.stats["vmax_mean"], 15.0))

        code, label, color = self.classify(vmax)
        return {
            "wind_speed": round(vmax, 1),        # knots
            "wind_kmh": round(vmax * 1.852, 1),
            "category": code,
            "lifecycle": label,
            "color": color,
            "approx_input": approx,              # True nếu PNG/JPG hoặc Kelvin
        }

    @staticmethod
    def classify(vmax_kt):
        for upper, code, label, color in CATEGORIES:
            if vmax_kt < upper:
                return code, label, color
        return CATEGORIES[-1][1:]

    @staticmethod
    def to_display(img_native):
        """Thang dataset (tối = lạnh) -> ảnh PIL kiểu vệ tinh (trắng = lạnh) để hiển thị."""
        g = 255.0 - np.clip(np.nan_to_num(img_native, nan=255.0), 0, 255)
        return Image.fromarray(g.astype(np.uint8))

    # giữ tên cũ cho app.py
    def bt_to_display(self, img_native):
        return self.to_display(img_native)


# ---------------------------------------------------------------------------
# Track model integration (lazy-loaded)
# ---------------------------------------------------------------------------
from pathlib import Path as _Path
from source.models.track_model import TrackPredictor as _TrackPredictor

DEFAULT_TRACK_PATHS = [
    "source/models/track_lgbm.pkl",
    "models/track_lgbm.pkl",
    "outputs/track_lgbm.pkl",
    "track_lgbm.pkl",
]

def _find_track_path():
    for p in DEFAULT_TRACK_PATHS:
        if _Path(p).exists():
            return p
    return None

def _get_track(self):
    if getattr(self, "_track", None) is None:
        path = getattr(self, "track_model_path", None) or _find_track_path()
        try:
            self._track = _TrackPredictor(path)
        except Exception as e:
            print(f"[predictor] Track model load failed ({e}); using persistence fallback")
            self._track = _TrackPredictor(None)
    return self._track

IntensityPredictor.track = property(_get_track)
