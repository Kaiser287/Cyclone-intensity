# File: src/training/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import pathlib
import os

# Import preprocessor để dùng chung logic chuẩn hóa ảnh
try:
    from source.data import preprocessor
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from source.data import preprocessor

class CycloneDataset(Dataset):
    """
    Class này chịu trách nhiệm:
    1. Tìm file ảnh trong máy tính.
    2. Đọc file ảnh lên.
    3. Lấy nhãn (sức gió) từ tên thư mục.
    """
    def __init__(self, root_dir, phase='train', img_size=224):
        self.root_dir = pathlib.Path(root_dir)
        self.phase = phase
        self.img_size = img_size
        
        # Quét tất cả file .jpg trong các thư mục con
        self.image_paths = list(self.root_dir.glob('*/*.jpg'))
        
        # Fallback: Nếu không tìm thấy .jpg thì tìm .png hoặc .jpeg
        if len(self.image_paths) == 0:
            self.image_paths = list(self.root_dir.glob('*/*.png')) + list(self.root_dir.glob('*/*.jpeg'))

        # Mặc định dùng transform từ preprocessor
        # (Lát nữa main.py có thể ghi đè cái này, nhưng cứ khai báo sẵn cho chắc)
        self.transform = preprocessor.get_transforms(phase=phase, img_size=img_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 1. Mở ảnh và convert sang RGB (quan trọng vì ResNet cần 3 kênh)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Nếu ảnh lỗi, tạo ảnh đen để tránh crash chương trình
            image = Image.new('RGB', (self.img_size, self.img_size))

        # 2. Logic lấy nhãn từ tên thư mục cha
        # Ví dụ: data/Cyclones/45/bao1.jpg -> Folder cha là "45" -> Nhãn = 45.0
        try:
            label = float(img_path.parent.name)
        except ValueError:
            label = 0.0 # Nếu tên folder không phải số (VD: "unknown"), gán là 0
            
        # 3. Áp dụng transform (Resize, Normalize...)
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)