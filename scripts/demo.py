# scripts/demo.py
# ============================================================================
# CYCLONE PREDICTION DEMO SCRIPT
# ============================================================================

import os
import sys
import time
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- SETUP ĐƯỜNG DẪN ĐỂ IMPORT ĐƯỢC MODULE TỪ SRC ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

# IMPORT CÁC MODULE TỪ DỰ ÁN (CẤU TRÚC SRC)
try:
    from source.models.genesis_model import GenesisClassifier
    from source.models.intensity_model import IntensityRegressionModel
    from source.models.track_model import TrackSeqModel
    from source.inference.predictor import GenesisPredictor, IntensityPredictor, TrackPredictor
except ImportError as e:
    print(f"Lỗi Import: {e}")
    print("Hãy chắc chắn bạn đang chạy từ thư mục gốc hoặc setup PYTHONPATH đúng.")

# ============================================================================
# UTILS & COLORS
# ============================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║             🌀 TYPHOON AI PREDICTION SYSTEM 🌀           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")

def print_section(title, icon="🔹"):
    print(f"\n{Colors.BOLD}{Colors.BLUE}=== {icon} {title.upper()} ==={Colors.ENDC}")

def print_info(msg): print(f"{Colors.CYAN}ℹ {msg}{Colors.ENDC}")
def print_success(msg): print(f"{Colors.GREEN}✓ {msg}{Colors.ENDC}")
def print_warning(msg): print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")
def print_result(label, value, unit=""):
    print(f"  {Colors.BOLD}{label}:{Colors.ENDC} {Colors.GREEN}{value}{Colors.ENDC} {unit}")

# ============================================================================
# DUMMY GENERATORS (TẠO DỮ LIỆU GIẢ LẬP)
# ============================================================================
def generate_dummy_satellite_image(size=224, seed=42):
    np.random.seed(seed)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    # Tạo hình xoắn ốc giả
    spiral = np.sin(5 * np.arctan2(Y, X) + 5 * R)
    image = np.exp(-R**2 / 0.5) * 0.5 + spiral * 0.1 + np.random.randn(size, size) * 0.05
    return np.clip(image, 0, 1).astype(np.float32)

def generate_dummy_sequence(num_frames=10, size=224, seed=42):
    return np.random.rand(num_frames, size, size).astype(np.float32)

def generate_dummy_track_labels(num_frames=10, seed=42):
    # Tạo quỹ đạo giả từ Đông Nam lên Tây Bắc
    lats = np.linspace(10, 20, num_frames)
    lons = np.linspace(115, 105, num_frames)
    return np.column_stack((lats, lons))

def load_image_input(input_path):
    """Hàm đọc ảnh (hỗ trợ cả .npy và .jpg/.png)"""
    if input_path.endswith('.npy'):
        return np.load(input_path)
    else:
        # Đọc ảnh JPG/PNG, convert sang grayscale, resize và normalize
        try:
            img = Image.open(input_path).convert('L') # Convert sang ảnh xám
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            return img_array.astype(np.float32)
        except Exception as e:
            print_warning(f"Không đọc được ảnh {input_path}: {e}")
            return generate_dummy_satellite_image()

# ============================================================================
# PLOTTING FUNCTIONS (VẼ BIỂU ĐỒ)
# ============================================================================
def plot_genesis_result(probability, save_path=None):
    # Placeholder: Vẽ đơn giản để code không quá dài
    plt.figure(figsize=(6, 2))
    plt.barh(['Genesis'], [probability], color='red' if probability > 0.5 else 'green')
    plt.xlim(0, 1)
    plt.title(f"Genesis Probability: {probability:.2%}")
    if save_path: plt.savefig(save_path)
    # plt.show() # Tạm tắt để chạy test nhanh

def plot_intensity_result(intensity, save_path=None):
    plt.figure(figsize=(6, 2))
    plt.barh(['Wind (kts)'], [intensity], color='orange')
    plt.xlim(0, 160)
    plt.title(f"Intensity: {intensity:.2f} kts")
    if save_path: plt.savefig(save_path)
    # plt.show()

def plot_track_result(predicted, true_track=None, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.plot(predicted[:, 1], predicted[:, 0], 'r-o', label='Predicted')
    if true_track is not None:
        plt.plot(true_track[:, 1], true_track[:, 0], 'b--x', label='True')
    plt.legend()
    plt.title("Track Prediction")
    if save_path: plt.savefig(save_path)
    # plt.show()

# ============================================================================
# CORE DEMO FUNCTIONS (CÁC HÀM CHÍNH BỊ THIẾU)
# ============================================================================

def demo_genesis(checkpoint_path=None, use_dummy=True, input_path=None):
    """Demo 1: Dự đoán hình thành bão"""
    print_section("GENESIS CLASSIFICATION", "🌀")
    
    # 1. Load Data
    if use_dummy or not input_path:
        image = generate_dummy_satellite_image()
    else:
        image = load_image_input(input_path)

    # 2. Prediction Logic
    if checkpoint_path and os.path.exists(checkpoint_path):
        # ... Code load model thật ở đây (giống logic cũ) ...
        # Tạm thời giả lập để test luồng
        probability = 0.85
    else:
        time.sleep(0.5)
        probability = np.random.uniform(0, 1)

    prediction = 1 if probability > 0.5 else 0
    
    # 3. Output
    status = "GENESIS DETECTED" if prediction else "NO GENESIS"
    print_result("Probability", f"{probability:.2%}")
    print_result("Status", status)
    
    # plot_genesis_result(probability, "outputs/genesis.png")
    return probability, prediction


def demo_intensity(checkpoint_path=None, use_dummy=True, input_path=None):
    """Demo 2: Dự đoán cường độ"""
    print_section("INTENSITY REGRESSION", "🌪️")

    # 1. Load Data
    if use_dummy or not input_path:
        image = generate_dummy_satellite_image()
    else:
        image = load_image_input(input_path)

    # 2. Prediction Logic
    if checkpoint_path and os.path.exists(checkpoint_path):
        print_info(f"Loading checkpoint: {checkpoint_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            # Lưu ý: Model thật cần input 3 channels, demo dummy đang dùng 1 channel
            # Ta giả lập việc convert chiều
            if len(image.shape) == 2:
                image_tensor = np.stack([image]*3, axis=0) # [3, H, W]
            else:
                image_tensor = image
            
            # Khởi tạo model và predictor
            model = IntensityRegressionModel(backbone="resnet50", input_channels=3)
            predictor = IntensityPredictor(model, checkpoint_path, device)
            
            # Predictor trả về dict {'wind_speed': float, 'lifecycle': str}
            # Nhưng hàm predict của IntensityPredictor nhận vào path hoặc PIL Image
            # Nếu ta đưa numpy array, cần convert lại
            # Để đơn giản hóa cho demo thật:
            res = predictor.predict(Image.fromarray((image*255).astype('uint8')))
            intensity = res['wind_speed']
            category = res['lifecycle']
            
        except Exception as e:
            print_warning(f"Lỗi load model thật: {e}. Chuyển sang giả lập.")
            intensity = np.random.uniform(30, 150)
            category = "Simulated Storm"
    else:
        time.sleep(0.5)
        intensity = np.random.uniform(30, 150)
        
        if intensity < 34: category = "Tropical Depression"
        elif intensity < 63: category = "Tropical Storm"
        elif intensity < 95: category = "Typhoon"
        else: category = "Super Typhoon"

    # 3. Output
    print_result("Wind Speed", f"{intensity:.2f}", "kts")
    print_result("Lifecycle", category)
    
    # plot_intensity_result(intensity, "outputs/intensity.png")
    return intensity, category


def demo_track(checkpoint_path=None, use_dummy=True, input_path=None, num_frames=10):
    """Demo 3: Dự đoán đường đi"""
    print_section("TRACK PREDICTION", "🗺️")

    # 1. Load Data
    seq = generate_dummy_sequence(num_frames)
    true_track = generate_dummy_track_labels(num_frames)

    # 2. Prediction Logic
    time.sleep(0.5)
    # Giả lập: Dự đoán lệch một chút so với thực tế
    noise = np.random.randn(num_frames, 2) * 0.1
    predicted_track = true_track + noise

    # 3. Output
    start_pt = predicted_track[0]
    end_pt = predicted_track[-1]
    print_result("Start Point", f"{start_pt[0]:.2f}, {start_pt[1]:.2f}")
    print_result("End Point", f"{end_pt[0]:.2f}, {end_pt[1]:.2f}")
    
    # plot_track_result(predicted_track, true_track, "outputs/track.png")
    return predicted_track, true_track


def demo_full_pipeline(use_dummy=True):
    """Demo toàn bộ"""
    print_banner()
    print_info("Running Full Pipeline Demo...")
    
    res_gen = demo_genesis(use_dummy=use_dummy)
    
    if res_gen[1] == 1: # Nếu có bão
        res_int = demo_intensity(use_dummy=use_dummy)
        res_track = demo_track(use_dummy=use_dummy)
    else:
        print_warning("No genesis detected. Pipeline stopped.")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='full', choices=['genesis', 'intensity', 'track', 'full'])
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--dummy', action='store_true', default=True)
    args = parser.parse_args()

    # Tạo folder output để lưu ảnh
    os.makedirs("outputs", exist_ok=True)

    if args.task == 'genesis':
        demo_genesis(args.checkpoint, args.dummy, args.input)
    elif args.task == 'intensity':
        demo_intensity(args.checkpoint, args.dummy, args.input)
    elif args.task == 'track':
        demo_track(args.checkpoint, args.dummy, args.input)
    elif args.task == 'full':
        demo_full_pipeline(args.dummy)

if __name__ == "__main__":
    main()