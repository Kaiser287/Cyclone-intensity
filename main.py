import os
import sys
import torch
from torch.utils.data import DataLoader, random_split

# --- 1. SETUP ĐƯỜNG DẪN ĐỂ IMPORT ĐƯỢC MODULE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from config import settings
    # Lưu ý: Import đúng tên folder là 'src'
    from source.training.dataset import CycloneDataset
    from source.models.intensity_model import IntensityRegressionModel
    from source.training.trainer import Trainer
    from source.data import preprocessor
except ImportError as e:
    print("❌ LỖI IMPORT: Không tìm thấy module.")
    print(f"Chi tiết: {e}")
    print("👉 Hãy chắc chắn tên folder là 'src' chứ không phải 'source'")
    sys.exit(1)

def main():
    print("\n" + "="*50)
    print(" 🌪️  HỆ THỐNG HUẤN LUYỆN BÃO (AUTO-RUN MODE)  🌪️")
    print("="*50)
    
    # --- 2. KIỂM TRA MÔI TRƯỜNG ---
    settings.print_config()
    
    # Kiểm tra folder dữ liệu có tồn tại không
    if not os.path.exists(settings.RAW_DATA_DIR):
        print(f"\n❌ LỖI NGHIÊM TRỌNG: Không tìm thấy folder dữ liệu!")
        print(f"👉 Đường dẫn cần tìm: {settings.RAW_DATA_DIR}")
        print("👉 Hãy bảo bạn của bạn giải nén bộ ảnh Kaggle vào đúng chỗ này.")
        return

    # --- 3. LOAD DATASET (Đọc trực tiếp từ ảnh JPG) ---
    print("\n[1/3] 📂 Đang quét ảnh từ ổ cứng...")
    try:
        # Load dataset train
        full_dataset = CycloneDataset(
            root_dir=settings.RAW_DATA_DIR, 
            phase='train',
            img_size=settings.INPUT_SIZE[0]
        )
        # Gán transform chuẩn
        full_dataset.transform = preprocessor.get_transforms(phase='train', img_size=settings.INPUT_SIZE[0])
        
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo Dataset: {e}")
        return

    if len(full_dataset) == 0:
        print("❌ Dataset rỗng! Hãy kiểm tra xem trong folder có ảnh .jpg không.")
        return

    # Chia tập Train/Val (80/20)
    val_size = int(settings.VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    # Gán transform riêng cho tập Val (chỉ resize, không xoay)
    val_data.dataset.transform = preprocessor.get_transforms(phase='val', img_size=settings.INPUT_SIZE[0])

    print(f"✅ Đã tải xong: {len(full_dataset)} ảnh.")
    print(f"   - Train: {len(train_data)}")
    print(f"   - Val:   {len(val_data)}")

    # Tạo DataLoader (Nơi bốc xếp dữ liệu lên GPU)
    train_loader = DataLoader(
        train_data, 
        batch_size=settings.BATCH_SIZE, 
        shuffle=True, 
        num_workers=settings.NUM_WORKERS,
        pin_memory=True if settings.DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=settings.BATCH_SIZE, 
        shuffle=False, 
        num_workers=settings.NUM_WORKERS,
        pin_memory=True if settings.DEVICE == 'cuda' else False
    )

    # --- 4. KHỞI TẠO MODEL (RESNET) ---
    print(f"\n[2/3] 🧠 Đang khởi tạo Model ({settings.MODEL_BACKBONE})...")
    try:
        model = IntensityRegressionModel(
            backbone=settings.MODEL_BACKBONE,
            input_channels=3, 
            pretrained=True
        )
        model = model.to(settings.DEVICE)
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Model: {e}")
        return

    # --- 5. BẮT ĐẦU TRAIN ---
    print("\n[3/3] 🔥 Bắt đầu quá trình huấn luyện...")
    print(f"👉 Tiến trình sẽ chạy trong {settings.NUM_EPOCHS} epochs.")
    
    # Gọi Class Trainer mà ta vừa sửa
    trainer = Trainer(model, train_loader, val_loader, task="regression")
    
    try:
        trainer.train()
        print("\n🎉 HOÀN TẤT XUẤT SẮC!")
        print(f"👉 File model đã được lưu tại: {settings.OUTPUTS_DIR}/best_model.pth")
        print("👉 Hãy gửi file đó lại cho chủ dự án nhé!")
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng thủ công.")
    except Exception as e:
        print(f"\n❌ Có lỗi trong lúc train: {e}")

if __name__ == "__main__":
    main()