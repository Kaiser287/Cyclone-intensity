import numpy as np

# Đường dẫn file nhãn
LABEL_PATH = "./data_source/Cyclone_Labels h5.npy"

try:
    labels = np.load(LABEL_PATH, allow_pickle=True)
    print("✅ Đã đọc được file!")
    print(f"👉 Kích thước: {labels.shape}")
    print("👉 Dữ liệu của mẫu đầu tiên (labels[0]):")
    print(labels[0])
    print("-" * 20)
    print("👉 Dữ liệu của mẫu thứ hai (labels[1]):")
    print(labels[1])
except Exception as e:
    print(f"❌ Lỗi: {e}")