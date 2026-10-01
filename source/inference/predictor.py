"""
Intensity predictor cho model ResNet18 (1 kênh IR) train trên TCIR (WPAC).

Đầu vào hợp lệ:
  - .npy / np.ndarray float: nhiệt độ sáng IR1 (Kelvin), giống dữ liệu train -> chính xác nhất
  - ảnh PNG/JPG grayscale: được quy đổi XẤP XỈ pixel -> Kelvin (xem PIXEL_TO_BT_*)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP = 128

# Thống kê từ tập train (dùng nếu checkpoint không lưu sẵn)
DEFAULT_STATS = {"img_mean": 250.75, "img_std": 30.20,
                 "vmax_mean": 53.91, "vmax_std": 32.48}

# Quy đổi ảnh 8-bit -> Kelvin theo quy ước ảnh IR: trắng = lạnh (mây cao)
# pixel 255 -> 180 K, pixel 0 -> 310 K. Đây là xấp xỉ, không phải hiệu chuẩn thật.
PIXEL_TO_BT_COLD = 180.0
PIXEL_TO_BT_WARM = 310.0

# Thang Saffir-Simpson (knots) – cùng cách chia khi đánh giá test
CATEGORIES = [
    (34,  "TD",  "Tropical Depression", "#4CAF50"),
    (64,  "TS",  "Tropical Storm",      "#CDDC39"),
    (83,  "C1",  "Typhoon Cat 1",       "#FFC107"),
    (96,  "C2",  "Typhoon Cat 2",       "#FF9800"),
    (113, "C3",  "Typhoon Cat 3",       "#FF5722"),
    (137, "C4",  "Typhoon Cat 4",       "#F44336"),
    (999, "C5",  "Super Typhoon Cat 5", "#9C27B0"),
]


# ---------------------------------------------------------------- model
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _clean_keys(sd):
    out = {}
    for k, v in sd.items():
        for p in ("module.", "_orig_mod."):
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    # bỏ prefix wrapper nếu toàn bộ key cùng prefix (vd "backbone.")
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
        # head dạng Sequential, vd [Dropout, Linear] hoặc [Linear, ReLU, Dropout, Linear]
        lin_idx = sorted({int(k.split(".")[1]) for k in sd
                          if k.startswith("fc.") and k.endswith(".weight")})
        layers = []
        for i in range(max(lin_idx) + 1):
            if i in lin_idx:
                w = sd[f"fc.{i}.weight"]
                layers.append(nn.Linear(w.shape[1], w.shape[0]))
            else:
                layers.append(nn.ReLU())  # Dropout = identity khi eval; ReLU lặp lại vô hại
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
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:  # torch cũ không có weights_only
            ckpt = torch.load(checkpoint_path, map_location=device)

        sd = _clean_keys(_extract_state_dict(ckpt))
        self.model = build_resnet18(sd)
        try:
            self.model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            sample = list(sd.keys())[:8]
            raise RuntimeError(
                f"Kiến trúc không khớp checkpoint. 8 key đầu: {sample}\n{e}"
            )
        self.model.to(device).eval()

        # Ưu tiên stats lưu trong checkpoint, nếu không có thì dùng mặc định
        self.stats = dict(DEFAULT_STATS)
        if isinstance(ckpt, dict):
            src = ckpt.get("stats", ckpt)
            for k in DEFAULT_STATS:
                if isinstance(src, dict) and k in src:
                    self.stats[k] = _scalar(src[k])

    # ---------- input -> Kelvin array (H, W)
    @staticmethod
    def pixels_to_bt(gray_uint8):
        g = gray_uint8.astype(np.float32) / 255.0
        return PIXEL_TO_BT_WARM - g * (PIXEL_TO_BT_WARM - PIXEL_TO_BT_COLD)

    def to_bt(self, x):
        """Trả về (bt_array, is_approx)."""
        if isinstance(x, str):
            if x.lower().endswith(".npy"):
                x = np.load(x)
            else:
                x = Image.open(x)
        if isinstance(x, Image.Image):
            return self.pixels_to_bt(np.array(x.convert("L"))), True

        arr = np.asarray(x)
        if arr.ndim == 3:  # (H, W, C) hoặc (C, H, W) -> lấy kênh IR đầu tiên
            arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
        if arr.dtype == np.uint8:
            return self.pixels_to_bt(arr), True
        return arr.astype(np.float32), False

    def _prepare(self, bt):
        bt = np.nan_to_num(bt, nan=self.stats["img_mean"])
        h, w = bt.shape
        t = torch.from_numpy(bt).float()[None, None]
        if h >= CROP and w >= CROP and abs(h - w) <= 2 and h <= 260:
            # ảnh TCIR gốc (~201x201): center crop như lúc train
            top, left = (h - CROP) // 2, (w - CROP) // 2
            t = t[..., top:top + CROP, left:left + CROP]
        else:
            t = F.interpolate(t, size=(CROP, CROP), mode="bilinear", align_corners=False)
        t = (t - self.stats["img_mean"]) / self.stats["img_std"]
        return t.to(self.device)

    # ---------- inference
    @torch.no_grad()
    def predict(self, x, tta=True):
        bt, approx = self.to_bt(x)
        t = self._prepare(bt)
        batch = torch.cat([torch.rot90(t, k, dims=(2, 3)) for k in range(4)]) if tta else t
        out = self.model(batch).float().view(-1).mean().item()
        vmax = out * self.stats["vmax_std"] + self.stats["vmax_mean"]
        vmax = float(max(vmax, 15.0))

        code, label, color = self.classify(vmax)
        return {
            "wind_speed": round(vmax, 1),          # knots
            "wind_kmh": round(vmax * 1.852, 1),
            "category": code,
            "lifecycle": label,
            "color": color,
            "approx_input": approx,                # True nếu ảnh PNG/JPG
        }

    @staticmethod
    def classify(vmax_kt):
        for upper, code, label, color in CATEGORIES:
            if vmax_kt < upper:
                return code, label, color
        return CATEGORIES[-1][1:]

    @staticmethod
    def bt_to_display(bt):
        """Kelvin -> ảnh PIL để hiển thị (trắng = lạnh)."""
        g = (PIXEL_TO_BT_WARM - np.nan_to_num(bt, nan=PIXEL_TO_BT_WARM)) / (
            PIXEL_TO_BT_WARM - PIXEL_TO_BT_COLD)
        return Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8))
