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