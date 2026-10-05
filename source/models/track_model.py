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

# ================== Track forecast (LightGBM) ==================
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

HORIZONS = (24, 48, 72)
FEATURES = ["lat0", "lon0", "dlat_24", "dlon_24", "dlat_48", "dlon_48", "vmax"]


def build_features(lat_m48, lon_m48, lat_m24, lon_m24, lat0, lon0, vmax):
    """Dùng chung cho cả lúc train và lúc inference để feature luôn khớp nhau."""
    lon_m48, lon_m24, lon0 = lon_m48 % 360, lon_m24 % 360, lon0 % 360
    row = [lat0, lon0,
           lat0 - lat_m24, lon0 - lon_m24,        # chuyển động 24h gần nhất
           lat_m24 - lat_m48, lon_m24 - lon_m48,  # chuyển động 24h trước đó
           vmax]
    return pd.DataFrame([row], columns=FEATURES, dtype=np.float32)


class TrackPredictor:
    """LightGBM dự báo độ dời (dlat, dlon) ở +24/48/72h; nếu thiếu model thì dùng persistence."""

    def __init__(self, model_path="models/track_lgbm.pkl"):
        p = Path(model_path)
        self.models = joblib.load(p)["models"] if p.exists() else None

    @property
    def is_ai(self):
        return self.models is not None

    def predict(self, history, vmax):
        """history = [(lat,lon) -48h, (lat,lon) -24h, (lat,lon) hiện tại]"""
        (la48, lo48), (la24, lo24), (la0, lo0) = history
        X = build_features(la48, lo48, la24, lo24, la0, lo0, vmax)
        lo0 = lo0 % 360
        v_lat, v_lon = X["dlat_24"][0], X["dlon_24"][0]
        points = []
        for h in HORIZONS:
            if self.is_ai:
                dlat = float(self.models[f"lat_{h}"].predict(X)[0])
                dlon = float(self.models[f"lon_{h}"].predict(X)[0])
            else:
                dlat, dlon = v_lat * h / 24, v_lon * h / 24
            points.append({"hour": h, "lat": la0 + dlat, "lon": lo0 + dlon})
        return {"method": "LightGBM" if self.is_ai else "Persistence", "points": points}
