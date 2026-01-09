import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# --- CẤU HÌNH ---
INPUT_SIZE = (128, 128) 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BasePredictor:
    def __init__(self, model, checkpoint_path=None):
        self.device = DEVICE
        self.model = model.to(self.device)
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        self.model.eval()

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ Không tìm thấy checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ Load model thành công!")
        except Exception as e:
            print(f"❌ Lỗi load checkpoint: {e}")

    def preprocess(self, img_input):
        if isinstance(img_input, str):
            img = Image.open(img_input).convert('RGB')
        elif isinstance(img_input, np.ndarray):
            img = Image.fromarray(img_input).convert('RGB')
        else:
            img = img_input.convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img).unsqueeze(0).to(self.device)

    def predict(self, img):
        raise NotImplementedError

class IntensityPredictor(BasePredictor):
    def predict(self, img_input):
        # 1. GIỮ ẢNH GỐC ĐỂ TÍNH ĐỘ SÁNG
        if isinstance(img_input, str):
            original_img = Image.open(img_input).convert('L')
        elif isinstance(img_input, np.ndarray):
            original_img = Image.fromarray(img_input).convert('L')
        else:
            original_img = img_input.convert('L')

        # 2. AI DỰ ĐOÁN GIÓ
        img_tensor = self.preprocess(img_input)
        with torch.no_grad():
            output = self.model(img_tensor)
        
        raw_output = output.cpu().item()
        
        # [QUAN TRỌNG] Vẫn dùng Model cũ -> Giữ hệ số 2.84
        WIND_CORRECTION = 2.84 
        
        final_wind_speed = raw_output * WIND_CORRECTION
        final_wind_speed = max(10, min(final_wind_speed, 185))

        # ==========================================================
        # 🌧️ CÔNG THỨC TÍNH MƯA (Đã tách riêng hệ số)
        # ==========================================================
        
        # Lấy độ sáng mây (0-255)
        pixel_array = np.array(original_img)
        cloud_brightness = np.percentile(pixel_array, 90) 
        
        # Tính mưa sơ bộ
        # 1. Mưa do gió (Wind): Gió 100kts -> Mưa ~50mm
        rain_wind = final_wind_speed * 0.5 
        
        # 2. Mưa do mây (Cloud): Mây trắng xoá -> Mưa ~80mm
        rain_cloud = (cloud_brightness / 255.0) * 100.0
        
        # Tổng hợp (Mây quan trọng hơn gió một chút)
        raw_rain = (rain_wind * 0.4) + (rain_cloud * 0.6)
        
        # [CHỈNH Ở ĐÂY] Hệ số chỉnh mưa riêng biệt
        # Nếu mưa đang quá BÉ -> Tăng lên 1.2, 1.5...
        # Nếu mưa đang quá TO -> Giảm xuống 0.8, 0.6...
        RAIN_SCALE = 1.5
        
        rainfall = raw_rain * RAIN_SCALE
        
        # Chặn giá trị hợp lý (5mm - 200mm)
        rainfall = max(5.0, min(rainfall, 200.0))
        # ==========================================================

        # Phân loại
        lifecycle_info = self.classify_vietnam_scale(final_wind_speed)
        
        return {
            "wind_speed": round(final_wind_speed, 2), 
            "rainfall": round(rainfall, 1),           
            "lifecycle": lifecycle_info['label'],     
            "color": lifecycle_info['color']
        }

    def classify_vietnam_scale(self, wind_knots):
        """
        Chuyển đổi Knots -> Km/h -> Cấp bão Việt Nam
        """
        kmh = wind_knots * 1.852
        
        if kmh < 62:
            return {"label": "ÁP THẤP NHIỆT ĐỚI (Cấp 6-7)", "color": "#008000"} 
        elif 62 <= kmh <= 88:
            return {"label": "BÃO THƯỜNG (Cấp 8-9)", "color": "#CCCC00"} 
        elif 89 <= kmh <= 117:
            return {"label": "BÃO MẠNH (Cấp 10-11)", "color": "#FF8C00"} 
        elif 118 <= kmh <= 133:
            return {"label": "BÃO RẤT MẠNH (Cấp 12)", "color": "#FF4500"} 
        elif 134 <= kmh <= 166:
            return {"label": "BÃO RẤT MẠNH (Cấp 13-14)", "color": "#FF0000"} 
        elif 167 <= kmh <= 183:
            return {"label": "BÃO DỮ DỘI (Cấp 15)", "color": "#8B0000"} 
        else:
            return {"label": "SIÊU BÃO (Trên cấp 16)", "color": "#800080"}