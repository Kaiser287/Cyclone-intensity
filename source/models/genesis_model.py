import torch
import torch.nn as nn
import torchvision.models as models

class GenesisClassifier(nn.Module):
    """
    Mô hình phân loại nhị phân: Dự đoán hình thành bão (Genesis vs No-Genesis).
    Hỗ trợ input Grayscale (ảnh hồng ngoại) hoặc RGB.
    """
    def __init__(self, backbone="resnet18", input_channels=1, pretrained=True):
        super().__init__()
        
        # --- 1. KHỞI TẠO BACKBONE (CÚ PHÁP MỚI) ---
        weights = None
        
        if backbone == "resnet18":
            if pretrained: weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
            
        elif backbone == "resnet50":
            if pretrained: weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.backbone = models.resnet50(weights=weights)
            
        else:
            # Mặc định fallback về resnet18 nếu nhập sai
            if pretrained: weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)

        # --- 2. XỬ LÝ INPUT CHANNELS (QUAN TRỌNG CHO ẢNH VỆ TINH) ---
        # Ảnh vệ tinh hồng ngoại thường chỉ có 1 kênh (Grayscale)
        # ResNet gốc nhận 3 kênh (RGB). Ta phải sửa layer đầu tiên.
        if input_channels != 3:
            original_conv1 = self.backbone.conv1
            
            # Tạo layer Conv2d mới với số kênh input đúng yêu cầu
            self.backbone.conv1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=False
            )
            
            # (Nâng cao) Copy trọng số trung bình từ RGB sang Grayscale để tận dụng Pretrained
            # Giúp model hội tụ nhanh hơn là random init
            if pretrained:
                with torch.no_grad():
                    self.backbone.conv1.weight[:] = torch.mean(original_conv1.weight, dim=1, keepdim=True)

        # --- 3. LAYER ĐẦU RA (BINARY CLASSIFICATION) ---
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        """
        x: [Batch, Channel, Height, Width]
        Return: [Batch] (Logits)
        """
        x = self.backbone(x)
        return x.squeeze(-1) # Output ra shape [Batch] thay vì [Batch, 1]