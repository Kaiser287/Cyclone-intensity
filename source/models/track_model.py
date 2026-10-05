import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    """
    Trích xuất đặc trưng ảnh từng bước từ chuỗi input.
    """
    def __init__(self, backbone="resnet18", input_channels=1, pretrained=True, feature_dim=256):
        super().__init__()
        
        # --- 1. SETUP BACKBONE (CHUẨN MỚI) ---
        weights = None
        if backbone == "resnet18":
            if pretrained: weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            if pretrained: weights = models.ResNet34_Weights.IMAGENET1K_V1
            resnet = models.resnet34(weights=weights)
        elif backbone == "resnet50":
            if pretrained: weights = models.ResNet50_Weights.IMAGENET1K_V1
            resnet = models.resnet50(weights=weights)
        else:
            # Fallback
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # --- 2. XỬ LÝ INPUT CHANNEL (CHO ẢNH VỆ TINH) ---
        if input_channels != 3:
            original_conv1 = resnet.conv1
            
            resnet.conv1 = nn.Conv2d(
                input_channels, 
                original_conv1.out_channels, 
                kernel_size=original_conv1.kernel_size, 
                stride=original_conv1.stride, 
                padding=original_conv1.padding, 
                bias=False
            )
            
            # [MẸO] Copy trọng số trung bình RGB -> Grayscale
            if pretrained:
                with torch.no_grad():
                    resnet.conv1.weight[:] = torch.mean(original_conv1.weight, dim=1, keepdim=True)

        # --- 3. FEATURE EXTRACTOR ---
        # Loại bỏ lớp FC cuối cùng (classifier)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]) # Output: [B, 512, 1, 1]
        self.flatten = nn.Flatten()
        
        # Projection layer: nén vector đặc trưng xuống kích thước mong muốn (VD: 256)
        self.proj = nn.Linear(resnet.fc.in_features, feature_dim)

    def forward(self, x):
        # x: [B * S, C, H, W] (Batch * Sequence gộp chung)
        feat = self.feature_extractor(x)
        feat = self.flatten(feat)
        return self.proj(feat)  # [B*S, feature_dim]


class TrackSeqModel(nn.Module):
    """
    Mô hình dự đoán quỹ đạo bão (chuỗi vị trí) dùng CNN encoder + LSTM decoder.
    Input: Chuỗi ảnh vệ tinh [Batch, Seq, Channel, Height, Width]
    Output: Chuỗi tọa độ (lat, lon) [Batch, Seq, 2]
    """
    def __init__(
        self,
        backbone="resnet18",
        input_channels=1,
        feature_dim=256,
        lstm_hidden=128,
        lstm_layers=1,
        output_dim=2 # (lat, lon)
    ):
        super().__init__()
        
        # CNN Encoder dùng chung weights
        self.cnn_encoder = CNNEncoder(
            backbone=backbone, 
            input_channels=input_channels, 
            pretrained=True, 
            feature_dim=feature_dim
        )
        
        # LSTM xử lý chuỗi thời gian
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # FC layer cuối ra tọa độ
        self.fc_out = nn.Linear(lstm_hidden, output_dim)

    def forward(self, x):
        """
        x: [B, S, C, H, W]
        """
        B, S, C, H, W = x.shape
        
        # 1. Gộp Batch và Sequence để đưa qua CNN (vì CNN chỉ nhận ảnh 2D)
        # [B, S, C, H, W] -> [B*S, C, H, W]
        x_reshaped = x.view(B*S, C, H, W)
        
        # 2. Trích xuất đặc trưng ảnh
        feats = self.cnn_encoder(x_reshaped)  # [B*S, feature_dim]
        
        # 3. Trả lại chiều Sequence cho LSTM
        # [B*S, feature_dim] -> [B, S, feature_dim]
        feats = feats.view(B, S, -1) 
        
        # 4. Đưa qua LSTM
        lstm_out, _ = self.lstm(feats) # [B, S, lstm_hidden]
        
        # 5. Dự đoán tọa độ
        coords = self.fc_out(lstm_out) # [B, S, 2]
        
        return coords

# =====================================================================
# TRACK PREDICTION (LightGBM) — dùng bởi predictor.py và train_track.py
# =====================================================================
import os
import math
import pickle
import numpy as np

HORIZONS = [24, 48, 72]  # giờ dự báo
FEATURES = [
    "lat0", "lon0",          # vị trí hiện tại (lon 0..360)
    "dlat24", "dlon24",      # dịch chuyển -24h -> now
    "dlat_prev", "dlon_prev",# dịch chuyển -48h -> -24h
    "speed24",               # quãng đường 24h gần nhất (km)
    "vmax",                  # cường độ hiện tại (kt), có thể NaN
]


def _to360(lon):
    return float(lon) % 360.0


def _wrap_dlon(d):
    return (d + 180.0) % 360.0 - 180.0


def _parse_point(p):
    if isinstance(p, dict):
        return float(p["lat"]), float(p["lon"])
    return float(p[0]), float(p[1])


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_features(history, vmax=None):
    """history: 3 điểm theo thứ tự [-48h, -24h, now], mỗi điểm (lat, lon) hoặc {'lat','lon'}."""
    pts = [_parse_point(p) for p in history]
    if len(pts) < 3:
        raise ValueError("track_history cần đủ 3 điểm: -48h, -24h, now")
    (la48, lo48), (la24, lo24), (la0, lo0) = pts[-3:]
    lo48, lo24, lo0 = _to360(lo48), _to360(lo24), _to360(lo0)

    v = float(vmax) if vmax is not None and not (isinstance(vmax, float) and math.isnan(vmax)) else np.nan
    return {
        "lat0": la0,
        "lon0": lo0,
        "dlat24": la0 - la24,
        "dlon24": _wrap_dlon(lo0 - lo24),
        "dlat_prev": la24 - la48,
        "dlon_prev": _wrap_dlon(lo24 - lo48),
        "speed24": float(haversine_km(la24, lo24, la0, lo0)),
        "vmax": v,
    }


class TrackPredictor:
    """Dùng LightGBM nếu có track_lgbm.pkl, nếu không thì fallback Persistence (ngoại suy tuyến tính)."""

    def __init__(self, model_path=None):
        self.models = None
        self.cone_km = None
        self.method = "Persistence"
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    bundle = pickle.load(f)
                self.models = bundle["models"]
                self.cone_km = bundle.get("cone_km")
                self.method = "LightGBM"
            except Exception as e:
                print(f"[TrackPredictor] Không load được {model_path}: {e} -> dùng Persistence")

    def predict(self, history, vmax=None):
        f = build_features(history, vmax)
        lat0, lon0 = f["lat0"], f["lon0"]
        x = np.array([[f[k] for k in FEATURES]], dtype=float)

        points = []
        for h in HORIZONS:
            if self.models is not None:
                dlat = float(self.models[f"{h}_dlat"].predict(x)[0])
                dlon = float(self.models[f"{h}_dlon"].predict(x)[0])
            else:
                dlat = f["dlat24"] * h / 24.0
                dlon = f["dlon24"] * h / 24.0
            points.append({
                "hour": h,
                "lat": round(lat0 + dlat, 3),
                "lon": round(_to360(lon0 + dlon), 3),  # app.py tự đổi sang -180..180
            })

        return {
            "method": self.method,
            "origin": {"lat": lat0, "lon": lon0},
            "points": points,
            "cone_km": self.cone_km,
        }
