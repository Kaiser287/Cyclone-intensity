import os
import pandas as pd
import torch
import re
from PIL import Image
from tqdm import tqdm
from source.inference.predictor import IntensityPredictor
from source.models.intensity_model import IntensityRegressionModel

# --- CẤU HÌNH QUAN TRỌNG ---
# 1. Đường dẫn đến thư mục gốc chứa dữ liệu
# Cấu trúc mong đợi:
# data/test_data/
#    ├── 30kts/
#    │   ├── storm1.jpg
#    │   └── storm2.jpg
#    ├── 50kts/
#    │   └── ...
#    └── 100_knots/
#        └── ...
DATA_ROOT_FOLDER = 'data/raw/Cyclones'  # <--- SỬA ĐƯỜNG DẪN NÀY CHO ĐÚNG MÁY BẠN

# 2. Đường dẫn Model
MODEL_PATH = 'outputs/best_model.pth'

# 3. Giới hạn số lượng ảnh mỗi thư mục (Để chạy cho nhanh)
# Lấy 5 ảnh mỗi cấp độ gió để làm báo cáo là đẹp
MAX_IMAGES_PER_FOLDER = 5 

def extract_wind_from_folder(folder_path):
    """
    Lấy tên thư mục cha và trích xuất con số trong đó.
    Ví dụ: 'data/test/100kts' -> Lấy '100kts' -> Trả về 100.0
    """
    folder_name = os.path.basename(folder_path)
    match = re.search(r'\d+', folder_name)
    if match:
        return float(match.group())
    return None

def generate_auto_report():
    print("⏳ Đang khởi động Model AI...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Model
    try:
        model = IntensityRegressionModel(backbone='resnet50', input_channels=3, pretrained=False)
        predictor = IntensityPredictor(model, checkpoint_path=MODEL_PATH)
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return

    if not os.path.exists(DATA_ROOT_FOLDER):
        print(f"❌ Không tìm thấy thư mục gốc: {DATA_ROOT_FOLDER}")
        return

    results = []
    total_images_processed = 0

    print(f"🚀 Bắt đầu quét dữ liệu trong: {DATA_ROOT_FOLDER}")
    print(f"⚠️ Chế độ lấy mẫu: Tối đa {MAX_IMAGES_PER_FOLDER} ảnh/folder.\n")

    # Dùng os.walk để duyệt qua tất cả thư mục con
    for root, dirs, files in os.walk(DATA_ROOT_FOLDER):
        # Lấy nhãn gió từ tên thư mục hiện tại
        actual_wind = extract_wind_from_folder(root)
        
        if actual_wind is not None:
            # Lọc lấy file ảnh
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                continue
                
            # Chỉ lấy số lượng giới hạn để demo
            selected_files = image_files[:MAX_IMAGES_PER_FOLDER]
            
            print(f"📂 Đang xử lý folder '{os.path.basename(root)}' (Gió thật: {actual_wind} kts) - {len(selected_files)} ảnh...")
            
            for img_name in selected_files:
                img_path = os.path.join(root, img_name)
                
                try:
                    # AI Dự đoán
                    pred = predictor.predict(img_path)
                    predicted_wind = pred['wind_speed']
                    
                    # Tính sai số
                    error = abs(predicted_wind - actual_wind)
                    
                    results.append({
                        "Thư mục nguồn": os.path.basename(root),
                        "Tên file ảnh": img_name,
                        "Thực tế (Knots)": actual_wind,
                        "AI Dự đoán (Knots)": predicted_wind,
                        "Độ lệch (MAE)": round(error, 2),
                        "Cấp bão (AI)": pred['lifecycle']
                    })
                    total_images_processed += 1
                    
                except Exception as e:
                    print(f"   ⚠️ Lỗi file {img_name}: {e}")
        
    # Xuất Excel
    if results:
        df_result = pd.DataFrame(results)
        # Sắp xếp theo gió thực tế
        df_result = df_result.sort_values(by="Thực tế (Knots)")
        
        output_file = "Du_Lieu_Thuc_Nghiem_Tu_Dong.xlsx"
        df_result.to_excel(output_file, index=False)
        
        mae_avg = df_result["Độ lệch (MAE)"].mean()
        
        print("\n" + "="*50)
        print(f"✅ HOÀN TẤT! Đã xử lý tổng cộng {total_images_processed} ảnh.")
        print(f"📄 File báo cáo: {output_file}")
        print(f"📊 Sai số trung bình (MAE) của mẫu thử: {mae_avg:.2f} Knots")
        print("="*50)
    else:
        print("❌ Không tìm thấy thư mục nào có chứa số (VD: '100kts') hoặc không có ảnh.")

if __name__ == "__main__":
    generate_auto_report()