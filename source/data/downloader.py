import os
import sys

# --- AN TOÀN: Chỉ import boto3 nếu máy có cài ---
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

def download_from_s3(bucket, key, destination):
    """
    Tải file từ S3 (Chỉ hoạt động nếu có cài boto3 và có AWS Key)
    """
    if not HAS_BOTO3:
        print("⚠️ CẢNH BÁO: Chưa cài thư viện 'boto3'. Bỏ qua bước tải S3.")
        return

    s3 = boto3.client('s3')
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        s3.download_file(bucket, key, destination)
        print(f"✅ Đã tải {key} -> {destination}")
    except NoCredentialsError:
        print("❌ Lỗi: Không tìm thấy credentials AWS.")
    except Exception as e:
        print(f"❌ Lỗi tải file: {e}")

def check_local_data(data_dir):
    """
    Hàm này quan trọng hơn cho đêm nay:
    Kiểm tra xem dữ liệu Kaggle đã được giải nén đúng chỗ chưa.
    """
    if not os.path.exists(data_dir):
        print(f"❌ LỖI: Thư mục dữ liệu không tồn tại: {data_dir}")
        print("👉 Hãy tạo thư mục này và giải nén dữ liệu vào đó!")
        return False
    
    # Đếm số lượng ảnh jpg
    count = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                count += 1
    
    if count == 0:
        print(f"⚠️ CẢNH BÁO: Thư mục {data_dir} rỗng hoặc không có ảnh!")
        return False
    else:
        print(f"✅ DỮ LIỆU SẴN SÀNG: Tìm thấy {count} ảnh trong {data_dir}")
        return True

if __name__ == "__main__":
    # Test thử
    # Vì đêm nay chạy Local nên mình chỉ check folder thôi
    # Đường dẫn này phải khớp với config/settings.py
    
    # Giả sử chạy từ thư mục gốc project
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_path = os.path.join(base_dir, "data", "raw", "Cyclones")
    
    print(f"--- Kiểm tra dữ liệu Local ---")
    check_local_data(raw_data_path)
    
    # Phần S3 này để đây cho đẹp, không chạy
    # bucket = "your-bucket"
    # download_from_s3(bucket, "test.txt", "data/test.txt")