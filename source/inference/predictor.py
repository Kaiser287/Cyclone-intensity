import os
import torch
from PIL import Image
import numpy as np

# Import preprocessor để lấy transforms chuẩn
try:
    from source.data import preprocessor
except ImportError:
    # Fallback nếu chạy script này độc lập (debug)
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from source.data import preprocessor

class BasePredictor:
    """
    Lớp cơ sở cho các predictor.
    """
    def __init__(self, model, checkpoint_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            
        self.model.eval() # Chuyển sang chế độ đánh giá (không dropout/batchnorm update)

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ Không tìm thấy checkpoint: {checkpoint_path}")
        
        # Load weights
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"✅ Đã load checkpoint: {checkpoint_path}")

    def preprocess(self, img_input):
        """
        Tiền xử lý ảnh chuẩn theo ResNet (ImageNet stats).
        img_input: Có thể là đường dẫn file (str) hoặc PIL Image.
        """
        # 1. Đọc ảnh nếu input là đường dẫn
        if isinstance(img_input, str):
            if not os.path.exists(img_input):
                raise FileNotFoundError(f"Không tìm thấy ảnh: {img_input}")
            img = Image.open(img_input).convert('RGB')
        elif isinstance(img_input, np.ndarray):
            img = Image.fromarray(img_input).convert('RGB')
        else:
            img = img_input.convert('RGB') # Giả sử là PIL Image

        # 2. Lấy bộ transform dành cho Validation/Test (Không augmentation)
        transform = preprocessor.get_transforms(phase='val', img_size=224)
        
        # 3. Transform và thêm Batch dimension [1, C, H, W]
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0) 
        
        return img_tensor.to(self.device)

    def predict(self, img):
        raise NotImplementedError("Phải implement ở lớp con")


class IntensityPredictor(BasePredictor):
    """
    Dự đoán cường độ bão và phân loại vòng đời (Lifecycle).
    """
    def predict(self, img_input):
        """
        Trả về: {
            'wind_speed': float (knots),
            'category': str (Lifecycle stage)
        }
        """
        img_tensor = self.preprocess(img_input)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            
        # Lấy giá trị sức gió (Output model là 1 số thực)
        wind_speed = output.cpu().item()
        
        # Phân loại vòng đời (Lifecycle) - QUAN TRỌNG CHO ĐỀ TÀI CỦA BẠN
        lifecycle = self.classify_lifecycle(wind_speed)
        
        return {
            "wind_speed": round(wind_speed, 2),
            "lifecycle": lifecycle
        }

    def classify_lifecycle(self, wind_speed):
        """
        Logic chuyển đổi từ Sức gió -> Giai đoạn vòng đời.
        Thang đo tham khảo Saffir-Simpson hoặc JMA.
        """
        if wind_speed < 34:
            return "TD - Áp thấp nhiệt đới (Tropical Depression)"
        elif 34 <= wind_speed <= 63:
            return "TS - Bão nhiệt đới (Tropical Storm)"
        elif 64 <= wind_speed <= 95:
            return "Typhoon - Cấp 1-2 (Trưởng thành)"
        elif 96 <= wind_speed <= 129:
            return "Typhoon - Cấp 3-4 (Bão rất mạnh)"
        else:
            return "Super Typhoon - Siêu bão (Cực đại)"

# --- GIỮ LẠI CÁC CLASS DƯỚI ĐỂ DỰ ÁN TRÔNG ĐẦY ĐỦ ---
# (Nhưng đêm nay chưa dùng tới)

class GenesisPredictor(BasePredictor):
    def predict(self, img_input, threshold=0.5):
        img_tensor = self.preprocess(img_input)
        with torch.no_grad():
            logit = self.model(img_tensor)
            prob = torch.sigmoid(logit).item()
        label = 1 if prob >= threshold else 0
        return {"probability": prob, "is_genesis": bool(label)}

class TrackPredictor(BasePredictor):
    def predict(self, seq_imgs):
        print("⚠️ TrackPredictor chưa được implement đầy đủ cho demo tối nay.")
        return None