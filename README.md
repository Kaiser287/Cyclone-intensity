# 🌪️ DỰ ĐOÁN CƯỜNG ĐỘ BÃO (CYCLONE INTENSITY PREDICTION)

Dự án sử dụng Deep Learning (**ResNet50**) để dự đoán tốc độ gió của bão dựa trên ảnh vệ tinh hồng ngoại (IR).

---

## 💻 Yêu cầu hệ thống (Recommended)
Code này đã được tối ưu hóa cho cấu hình sau:
* **GPU:** NVIDIA RTX 4060 (Laptop/PC) trở lên.
* **RAM:** 16GB.
* **Python:** 3.8 - 3.10.
* **CUDA:** Đã cài đặt driver NVIDIA mới nhất.

---

## ⚙️ 1. Cài đặt môi trường
Mở Terminal tại thư mục dự án và chạy lệnh sau:

pip install torch torchvision numpy pandas pillow tqdm h5py matplotlib scikit-learn


---

## 📂 2. Tải Dữ liệu (QUAN TRỌNG)
Vì dữ liệu gốc rất nặng (~5GB) nên không được upload lên GitHub. Bạn cần tải thủ công:

**Bước 1:** Tải về từ Kaggle:
👉 **[Download TCIR Dataset](https://www.kaggle.com/datasets/vaukaofworlds/thecycloneimagedataset)**

**Bước 2:** Giải nén và lấy 2 file: `Cyclone_Images.h5` và `Cyclone_Labels.npy`.

**Bước 3:** Tại thư mục code, tạo folder `data_source` và bỏ 2 file đó vào:

Project_Folder/
├── config/
├── src/
├── data_source/               <-- TẠO MỚI
│   ├── Cyclone_Images.h5      <-- Bỏ vào đây
│   └── Cyclone_Labels.npy     <-- Bỏ vào đây
├── convert_h5_to_jpg.py
├── main.py
└── ...


---

## 🚀 3. Chạy Dự án

### Bước 1: Bung nén dữ liệu (Chạy 1 lần đầu)
Chạy lệnh sau để chuyển đổi file `.h5` sang ảnh `.jpg`:

python convert_h5_to_jpg.py

*Thời gian: 1-2 phút.*

### Bước 2: Huấn luyện (Training)
Sau khi có ảnh, chạy lệnh train:

python main.py

### ℹ️ Cấu hình (RTX 4060)
* **Batch Size:** 64
* **Epochs:** 50
* **Output:** Model lưu tại `outputs/best_model.pth`