import os
import torch

# ================== ĐƯỜNG DẪN (PATH) ==================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Trỏ đúng vào folder chứa ảnh đã bung (QUAN TRỌNG)
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "Cyclones")

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ================== TRAINING CONFIG (RTX 4060 LAPTOP) ==================
# RTX 4060 8GB VRAM cân tốt Batch 64 với ảnh 128x128
BATCH_SIZE = 64         
NUM_EPOCHS = 50         # Chạy 50 vòng (tầm 2-3 tiếng)
LEARNING_RATE = 0.0001
VAL_SPLIT = 0.2
SEED = 42
NUM_WORKERS = 2         # Windows để 2 là ổn định nhất

# ================== MODEL CONFIG ==================
MODEL_BACKBONE = "resnet50"
INPUT_SIZE = (128, 128) # Giữ nguyên theo dataset gốc
NUM_CLASSES = 1         

# ================== DEVICE ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== LOGGING ==================
LOG_FILE = os.path.join(BASE_DIR, "training.log")
VERBOSE = True

def print_config():
    print(f"\n[CẤU HÌNH RTX 4060] Device: {DEVICE} | Model: {MODEL_BACKBONE} | Batch: {BATCH_SIZE}")