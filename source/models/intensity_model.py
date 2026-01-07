import torch
import torch.nn as nn
import torchvision.models as models

class IntensityRegressionModel(nn.Module):
    """
    Mô hình hồi quy cường độ bão (Regression).
    Input: Ảnh vệ tinh [Batch, Channel, Height, Width]
    Output: Sức gió (Knots) [Batch]
    """
    def __init__(self, backbone="resnet50", input_channels=1, pretrained=True):
        super().__init__()
        
        # --- 1. KHỞI TẠO BACKBONE (CÚ PHÁP MỚI) ---
        # Mặc định dùng ResNet50 vì bạn có GPU 3060 mạnh
        weights = None
        
        if backbone == "resnet18":
            if pretrained: weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
            
        elif backbone == "resnet34":
            if pretrained: weights = models.ResNet34_Weights.IMAGENET1K_V1
            self.backbone = models.resnet34(weights=weights)
            
        elif backbone == "resnet50":
            if pretrained: weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.backbone = models.resnet50(weights=weights)
            
        else:
            raise ValueError(f"Backbone {backbone} chưa được hỗ trợ.")

        # --- 2. XỬ LÝ INPUT CHANNEL (CHO ẢNH VỆ TINH) ---
        if input_channels != 3:
            original_conv1 = self.backbone.conv1
            
            # Tạo layer Conv2d mới đè lên layer cũ
            self.backbone.conv1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=False
            )
            
            # [MẸO PRO] Copy trọng số từ RGB sang Grayscale
            # Giúp model không phải học lại từ đầu cái việc "nhìn cạnh/góc"
            if pretrained:
                with torch.no_grad():
                    # Lấy trung bình cộng 3 kênh RGB gán cho 1 kênh Gray
                    self.backbone.conv1.weight[:] = torch.mean(original_conv1.weight, dim=1, keepdim=True)

        # --- 3. LAYER OUTPUT (REGRESSION) ---
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        # x: [B, C, H, W] -> Output: [B, 1]
        x = self.backbone(x)
        
        # Squeeze để chuyển từ [Batch, 1] thành [Batch] cho khớp với label
        return x.squeeze(-1)