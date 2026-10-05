"""
Dự đoán quỹ đạo bão từ lịch sử vị trí (IBTrACS - Tây Bắc Thái Bình Dương).

- LightGBM : mỗi cặp (mốc giờ, thành phần dlat/dlon) là một regressor riêng.
- Persistence: ngoại suy tuyến tính vận tốc 24h gần nhất (fallback khi chưa có model).

File này được dùng chung bởi train_track.py, predictor.py và app.py để đặc trưng
lúc train và lúc chạy luôn khớp nhau.

Phần LightGBM KHÔNG cần torch. CNNEncoder / TrackSeqModel (bản cũ, dùng chuỗi ảnh)
được giữ ở cuối file để không làm hỏng code nào còn import chúng; chỉ được định
nghĩa khi môi trường có torch.

Đặt tại: source/models/track_model.py
"""
import os

import numpy as np

# =====================================================================
# TRACK PREDICTION (LightGBM) — dùng bởi predictor.py và train_track.py
# =====================================================================
HORIZONS = [12, 24, 36, 48]  # giờ dự báo

FEATURES = [
    "lat0", "lon0",
    "dlat_24", "dlon_24",   # dịch chuyển 24h gần nhất (độ; dlon đã nhân cos(lat))
    "dlat_48", "dlon_48",   # dịch chuyển 24h trước đó
    "acc_lat", "acc_lon",   # thay đổi vận tốc (gia tốc)
    "speed_kmh", "head_sin", "head_cos",
    "vmax",                 # cường độ hiện tại (kt) - từ CNN, có thể NaN
]

# Bán kính nón sai số (km) mặc định cho Persistence - giá trị XẤP XỈ.
# Khi có model LightGBM, giá trị đo thật trên tập validation sẽ được dùng.
PERSISTENCE_CONE_KM = {12: 80.0, 24: 160.0, 36: 250.0, 48: 350.0}

EARTH_R = 6371.0


# ---------------------------------------------------------------- helpers
def dlon_deg(a, b):
    """Hiệu kinh độ a - b, đưa về khoảng [-180, 180) (xử lý qua kinh tuyến 180)."""
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Khoảng cách vòng lớn (km). Chạy được với số hoặc numpy array."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _parse_point(p):
    """Nhận (lat, lon) hoặc {'lat':..., 'lon':...}."""
    if isinstance(p, dict):
        return float(p["lat"]), float(p["lon"])
    return float(p[0]), float(p[1])


def build_features(lat48, lon48, lat24, lon24, lat0, lon0, vmax=None):
    """Tạo ma trận đặc trưng (n, len(FEATURES)) theo đúng thứ tự FEATURES.
    Nhận số đơn hoặc numpy array (dùng cho cả train lẫn inference)."""
    lat48, lon48, lat24, lon24, lat0, lon0 = (
        np.asarray(v, dtype=float) for v in (lat48, lon48, lat24, lon24, lat0, lon0))
    vmax = np.asarray(np.nan if vmax is None else vmax, dtype=float)
    lon48, lon24, lon0 = lon48 % 360.0, lon24 % 360.0, lon0 % 360.0

    dlat_24 = lat0 - lat24
    dlon_24 = dlon_deg(lon0, lon24) * np.cos(np.radians(lat0))
    dlat_48 = lat24 - lat48
    dlon_48 = dlon_deg(lon24, lon48) * np.cos(np.radians(lat24))
    speed = haversine_km(lat24, lon24, lat0, lon0) / 24.0
    heading = np.arctan2(dlon_24, dlat_24)  # 0 = Bắc, pi/2 = Đông

    cols = [lat0, lon0, dlat_24, dlon_24, dlat_48, dlon_48,
            dlat_24 - dlat_48, dlon_24 - dlon_48,
            speed, np.sin(heading), np.cos(heading), vmax]
    cols = np.broadcast_arrays(*cols)
    return np.column_stack([np.atleast_1d(c).ravel() for c in cols])


# ---------------------------------------------------------------- predictor
class TrackPredictor:
    """
    history: 3 điểm theo thứ tự cũ -> mới: [-48h, -24h, hiện tại],
             mỗi điểm là (lat, lon) hoặc {'lat', 'lon'}.
    vmax   : cường độ hiện tại (kt), lấy từ model CNN.
    """

    def __init__(self, model_path=None):
        self.models = None
        self.horizons = list(HORIZONS)
        self.cone_km = dict(PERSISTENCE_CONE_KM)
        self.metrics = {}
        self.load_error = None

        if not model_path:
            return
        if not os.path.exists(model_path):
            self.load_error = f"Không tìm thấy {os.path.basename(model_path)}"
            return
        try:
            import joblib
            bundle = joblib.load(model_path)
            if list(bundle.get("features", [])) != FEATURES:
                raise ValueError("Danh sách đặc trưng trong model không khớp FEATURES hiện tại "
                                 "(model cũ? hãy train lại bằng train_track.py)")
            horizons = [int(h) for h in bundle.get("horizons", HORIZONS)]
            models = bundle["models"]
            missing = [(h, c) for h in horizons for c in ("dlat", "dlon") if (h, c) not in models]
            if missing:
                raise KeyError(f"Thiếu regressor cho {missing}")
            cone = {int(h): float(v) for h, v in bundle.get("cone_km", {}).items()}

            self.models, self.horizons = models, horizons
            self.cone_km = cone or self.cone_km
            self.metrics = bundle.get("metrics", {})
        except Exception as e:  # thiếu lightgbm, file hỏng, sai phiên bản...
            self.models = None
            self.load_error = f"{type(e).__name__}: {e}"
            print(f"[TrackPredictor] {self.load_error} -> dùng Persistence")

    @property
    def method(self):
        return "LightGBM" if self.models else "Persistence"

    @property
    def is_ai(self):
        return self.models is not None

    def predict(self, history, vmax=None):
        pts = [_parse_point(p) for p in history]
        if len(pts) != 3:
            raise ValueError("history cần đúng 3 điểm: [-48h, -24h, hiện tại]")
        (lat48, lon48), (lat24, lon24), (lat0, lon0) = pts

        points = []
        if self.is_ai:
            X = build_features(lat48, lon48, lat24, lon24, lat0, lon0, vmax)
            for h in self.horizons:
                dlat = float(self.models[(h, "dlat")].predict(X)[0])
                dlon = float(self.models[(h, "dlon")].predict(X)[0])
                points.append((h, lat0 + dlat, lon0 + dlon))
        else:
            v_lat = (lat0 - lat24) / 24.0
            v_lon = float(dlon_deg(lon0, lon24)) / 24.0
            for h in self.horizons:
                points.append((h, lat0 + v_lat * h, lon0 + v_lon * h))

        default_cone = PERSISTENCE_CONE_KM.get(48, 350.0)
        return {
            "method": self.method,
            "is_ai": self.is_ai,
            "origin": {"lat": lat0, "lon": lon0},
            "history": [{"hour": -48, "lat": lat48, "lon": lon48},
                        {"hour": -24, "lat": lat24, "lon": lon24},
                        {"hour": 0, "lat": lat0, "lon": lon0}],
            "points": [{"hour": h,
                        "lat": round(float(np.clip(la, -89.0, 89.0)), 2),
                        "lon": round(float(lo), 2),
                        "cone_km": round(self.cone_km.get(h, default_cone), 0)}
                       for h, la, lo in points],
            "cone_km": dict(self.cone_km),
        }


# =====================================================================
# LEGACY: CNN + LSTM dự đoán quỹ đạo từ chuỗi ảnh (không dùng trong app).
# Giữ lại để code cũ còn import không bị lỗi. Chỉ định nghĩa khi có torch.
# =====================================================================
try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if _HAS_TORCH:
    class CNNEncoder(nn.Module):
        """Trích xuất đặc trưng ảnh từng bước từ chuỗi input."""

        def __init__(self, backbone="resnet18", input_channels=1, pretrained=True, feature_dim=256):
            super().__init__()
            builders = {
                "resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights.IMAGENET1K_V1),
                "resnet34": (tv_models.resnet34, tv_models.ResNet34_Weights.IMAGENET1K_V1),
                "resnet50": (tv_models.resnet50, tv_models.ResNet50_Weights.IMAGENET1K_V1),
            }
            fn, w = builders.get(backbone, builders["resnet18"])
            resnet = fn(weights=w if pretrained else None)

            if input_channels != 3:
                old = resnet.conv1
                resnet.conv1 = nn.Conv2d(input_channels, old.out_channels,
                                         kernel_size=old.kernel_size, stride=old.stride,
                                         padding=old.padding, bias=False)
                if pretrained:
                    with torch.no_grad():
                        resnet.conv1.weight[:] = old.weight.mean(dim=1, keepdim=True)

            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            self.flatten = nn.Flatten()
            self.proj = nn.Linear(resnet.fc.in_features, feature_dim)

        def forward(self, x):  # x: [B*S, C, H, W]
            return self.proj(self.flatten(self.feature_extractor(x)))

    class TrackSeqModel(nn.Module):
        """CNN encoder + LSTM: chuỗi ảnh [B, S, C, H, W] -> tọa độ [B, S, 2]."""

        def __init__(self, backbone="resnet18", input_channels=1, feature_dim=256,
                     lstm_hidden=128, lstm_layers=1, output_dim=2):
            super().__init__()
            self.cnn_encoder = CNNEncoder(backbone, input_channels, True, feature_dim)
            self.lstm = nn.LSTM(feature_dim, lstm_hidden, lstm_layers, batch_first=True)
            self.fc_out = nn.Linear(lstm_hidden, output_dim)

        def forward(self, x):
            B, S, C, H, W = x.shape
            feats = self.cnn_encoder(x.reshape(B * S, C, H, W)).reshape(B, S, -1)
            out, _ = self.lstm(feats)
            return self.fc_out(out)
