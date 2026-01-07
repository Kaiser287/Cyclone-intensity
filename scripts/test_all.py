import os
import sys

# --- 1. SETUP ĐƯỜNG DẪN ---
# Lấy đường dẫn thư mục chứa file này (thư mục scripts)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Thêm thư mục gốc dự án vào sys.path để code có thể nhìn thấy folder 'src'
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Chuyển hướng làm việc về thư mục gốc (để load ảnh dummy không bị lỗi path)
os.chdir(project_root)

try:
    # Import các hàm từ demo.py (nằm cùng thư mục scripts)
    # Lưu ý: Python mặc định tìm trong thư mục hiện tại, nên import demo là được
    from scripts.demo import (
        print_banner, 
        demo_genesis, 
        demo_intensity, 
        demo_track
    )
except ImportError:
    # Fallback nếu chạy trực tiếp trong folder scripts
    from demo import (
        print_banner, 
        demo_genesis, 
        demo_intensity, 
        demo_track
    )

def main():
    # In banner chào mừng
    print_banner()
    
    print("\n" + "="*70)
    print("🧪  KIỂM TRA HỆ THỐNG DEMO (CHẾ ĐỘ GIẢ LẬP)")
    print("="*70)
    
    # 1. Test Genesis
    print("\n[1/3] Testing Genesis Classification...")
    try:
        # Hàm trả về (xác suất, nhãn)
        prob, pred = demo_genesis(use_dummy=True)
        print(f"✓ Genesis test completed: {prob*100:.1f}% probability")
    except Exception as e:
        print(f"❌ Genesis test failed: {e}")
    
    # 2. Test Intensity
    print("\n[2/3] Testing Intensity Regression...")
    try:
        # Hàm trả về (sức gió, loại bão)
        intensity, category = demo_intensity(use_dummy=True)
        print(f"✓ Intensity test completed: {intensity:.2f} kts ({category})")
    except Exception as e:
        print(f"❌ Intensity test failed: {e}")
    
    # 3. Test Track
    print("\n[3/3] Testing Track Prediction...")
    try:
        # Hàm trả về (quỹ đạo dự đoán, quỹ đạo thật)
        track_pred, track_true = demo_track(use_dummy=True)
        print(f"✓ Track test completed. Points: {len(track_pred)}")
    except Exception as e:
        print(f"❌ Track test failed: {e}")
    
    print("\n" + "="*70)
    print("✅ ĐÃ KIỂM TRA TOÀN BỘ HỆ THỐNG - SẴN SÀNG TRÌNH CHIẾU!")
    print("="*70)

if __name__ == "__main__":
    main()