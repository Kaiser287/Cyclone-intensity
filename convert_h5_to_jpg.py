import h5py
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# --- CẤU HÌNH ---
H5_PATH = "./data_source/Cyclone_Images.h5"
LABEL_PATH = "./data_source/Cyclone_Labels h5.npy"
OUTPUT_DIR = "./data/raw/Cyclones"

def convert_data():
    print("🚀 Đang khởi động bộ chuyển đổi dữ liệu...")
    
    # 1. Load Labels (Bắt buộc allow_pickle=True)
    try:
        labels = np.load(LABEL_PATH, allow_pickle=True)
        print(f"✅ Đã load Labels: {labels.shape}")
    except Exception as e:
        print(f"❌ Lỗi đọc Labels: {e}")
        return

    # 2. Load Images
    try:
        h5_file = h5py.File(H5_PATH, 'r')
        key = list(h5_file.keys())[0]
        images = h5_file[key]
        print(f"✅ Đã load Images: {images.shape}")
    except Exception as e:
        print(f"❌ Lỗi đọc H5: {e}")
        return

    num_samples = min(len(images), len(labels))
    print(f"📸 Bắt đầu xử lý {num_samples} mẫu...")
    
    count = 0
    # 3. Vòng lặp
    for i in tqdm(range(num_samples)):
        try:
            # === SỬA LỖI Ở ĐÂY: Lấy cột thứ 6 (index 5) là Tốc độ gió ===
            # Dữ liệu: ['ATLN' 'ID' Lon Lat Time Wind Speed Pressure]
            row_data = labels[i]
            wind_speed = float(row_data[5]) # Lấy giá trị 30.0, 45.0...
            
            # Làm tròn về bội số 5
            folder_speed = int(5 * round(wind_speed / 5))
            
            # === XỬ LÝ ẢNH ===
            img_data = images[i]
            
            # Chuyển kênh (128,128,4) -> (4,128,128) nếu cần
            # Dataset này thường là (Height, Width, Channel) -> OK
            
            # Lấy kênh IR (thường là kênh 0)
            ir_channel = img_data[:, :, 0]
            
            # Chuẩn hóa
            if ir_channel.max() <= 1.0:
                ir_channel = (ir_channel * 255).astype(np.uint8)
            else:
                ir_channel = np.clip(ir_channel, 0, 255).astype(np.uint8)
                
            # === LƯU FILE ===
            save_dir = os.path.join(OUTPUT_DIR, str(folder_speed))
            os.makedirs(save_dir, exist_ok=True)
            
            # Tên file chứa luôn thông tin để sau này dễ check
            # VD: storm_0_30kt.jpg
            file_name = f"storm_{i}_{folder_speed}kt.jpg"
            
            img = Image.fromarray(ir_channel, mode='L')
            img.save(os.path.join(save_dir, file_name))
            count += 1
            
        except Exception as e:
            continue

    print(f"\n🎉 THÀNH CÔNG RỰC RỠ! Đã bung {count} ảnh vào '{OUTPUT_DIR}'")
    print("👉 Bây giờ hãy chạy lệnh: python main.py")

if __name__ == "__main__":
    convert_data()