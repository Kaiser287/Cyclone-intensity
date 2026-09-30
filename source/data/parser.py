import os
import sys

# --- AN TOÀN: Chỉ import xarray nếu máy có cài đặt ---
# Giúp bạn của bạn không bị lỗi "ModuleNotFoundError" nếu chưa cài thư viện nặng này
try:
    import xarray as xr
    import numpy as np
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

def parse_netcdf(file_path, variables=None):
    """
    Đọc file NetCDF (.nc).
    LƯU Ý: Đêm nay chạy dữ liệu ảnh JPG nên hàm này sẽ không được gọi.
    """
    if not HAS_XARRAY:
        print("⚠️ CẢNH BÁO: Chưa cài thư viện 'xarray'. Không thể đọc file NetCDF.")
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    
    try:
        ds = xr.open_dataset(file_path)
        if variables:
            # Chỉ lấy các biến cần thiết
            ds = ds[variables]
        return ds
    except Exception as e:
        print(f"❌ Lỗi khi đọc file NetCDF: {e}")
        return None

def get_variable_as_numpy(ds, var_name):
    """
    Trích xuất biến thành numpy array.
    """
    if ds is None:
        return None
        
    if var_name not in ds:
        raise KeyError(f"Biến '{var_name}' không tồn tại trong dataset")
    
    return ds[var_name].values

def get_metadata(ds):
    """
    Lấy metadata.
    """
    if ds is None:
        return {}
    return dict(ds.attrs)

# Ví dụ test (Chỉ chạy khi gọi trực tiếp file này)
if __name__ == "__main__":
    if HAS_XARRAY:
        print("✅ Đã cài đặt xarray. Sẵn sàng xử lý file .nc (nếu có).")
    else:
        print("⚠️ Chưa cài đặt xarray. File này đang ở chế độ chờ (Safe Mode).")