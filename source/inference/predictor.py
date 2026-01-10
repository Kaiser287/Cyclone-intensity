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
            raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")

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
    
    # === [HÀM MỚI] HIỆU CHỈNH THÔNG MINH ===
    def calibrate_smart(self, raw_val):
        """
        Hàm này giúp chỉnh riêng bão Trung bình mà không ảnh hưởng bão To.
        """
        # 1. Bão Cận Siêu Bão & Siêu Bão (> 95 Knots)
        # Bạn bảo đoạn này đã chuẩn -> Giữ nguyên (Hệ số 1.0)
        if raw_val >= 95:
            return raw_val * 1.0 
        
        # 2. Bão Trung Bình & Mạnh (60 - 95 Knots)
        # Đoạn này "chưa oke lắm". Thường model sẽ đoán thấp hơn thực tế.
        # -> Thử nhân nhẹ lên 1.15 hoặc 1.2 xem sao.
        elif 60 <= raw_val < 95:
            return raw_val * 2.84  # <--- CHỈNH SỐ NÀY (Tăng lên nếu muốn bão to hơn)
            
        # 3. Bão Yếu & Áp thấp (< 60 Knots)
        # Có thể cần nhân mạnh hơn chút vì mây rất loãng
        else:
            return raw_val * 2.5   # <--- CHỈNH SỐ NÀY
            
    def predict(self, img_input):
        # 1. Xử lý ảnh
        if isinstance(img_input, str):
            original_img = Image.open(img_input).convert('L')
        elif isinstance(img_input, np.ndarray):
            original_img = Image.fromarray(img_input).convert('L')
        else:
            original_img = img_input.convert('L')

        # 2. AI Dự đoán (Ra giá trị thô)
        img_tensor = self.preprocess(img_input)
        with torch.no_grad():
            output = self.model(img_tensor)
        
        raw_output = output.cpu().item()
        
        # 3. ÁP DỤNG HIỆU CHỈNH THÔNG MINH
        # Thay vì nhân một số cố định, ta gọi hàm calibrate_smart
        final_wind_speed = self.calibrate_smart(raw_output)
        
        # Chặn trần/sàn cho hợp lý
        final_wind_speed = max(15, min(final_wind_speed, 185))

        # 4. TÍNH LƯỢNG MƯA
        pixel_array = np.array(original_img)
        cloud_brightness = np.percentile(pixel_array, 90) 
        
        rain_from_wind = final_wind_speed * 0.5 
        rain_from_cloud = (cloud_brightness / 255.0) * 100.0
        
        rainfall = (rain_from_wind * 0.4) + (rain_from_cloud * 0.6)
        rainfall = max(5.0, min(rainfall * 1.0, 200.0))

        # 5. PHÂN LOẠI (English Interface)
        lifecycle_info = self.classify_international_scale(final_wind_speed)
        
        return {
            "wind_speed": round(final_wind_speed, 2), 
            "rainfall": round(rainfall, 1),           
            "lifecycle": lifecycle_info['label'],     
            "color": lifecycle_info['color']
        }

    def classify_international_scale(self, wind_knots):
        kmh = wind_knots * 1.852
        kmh = round(kmh, 1)

        if kmh < 62:
            return {"label": "TROPICAL DEPRESSION (TD)", "color": "#008000"} 
        elif 62 <= kmh <= 88:
            return {"label": "TROPICAL STORM (TS)", "color": "#CCCC00"} 
        elif 89 <= kmh <= 117:
            return {"label": "SEVERE TROPICAL STORM", "color": "#FF8C00"} 
        elif 118 <= kmh <= 156:
            return {"label": "TYPHOON (Cat 1-2)", "color": "#FF4500"} 
        elif 157 <= kmh <= 183:
            return {"label": "VERY STRONG TYPHOON (Cat 3-4)", "color": "#FF0000"} 
        else:
            return {"label": "SUPER TYPHOON (Cat 5+)", "color": "#800080"}