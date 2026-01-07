import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np

# Import config để lấy các tham số
try:
    from config import settings
except ImportError:
    # Fallback nếu chạy riêng lẻ
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from config import settings

class Trainer:
    """
    Class quản lý quy trình huấn luyện.
    Hỗ trợ cả Regression (Cường độ) và Classification (Genesis).
    """
    def __init__(self, model, train_loader, val_loader, task="regression"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task = task # 'regression' hoặc 'classification'
        self.device = settings.DEVICE
        
        # Định nghĩa hàm Loss và Optimizer
        if self.task == "classification":
            self.criterion = nn.BCEWithLogitsLoss() # Cho nhị phân
        else:
            self.criterion = nn.MSELoss() # Cho hồi quy (Cường độ)
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=settings.LEARNING_RATE)
        self.best_metric = float('inf') if task == 'regression' else 0.0

    def train_one_epoch(self, epoch_index):
        self.model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []

        # Thanh progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_index} [TRAIN]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device) # Shape [Batch]

            self.optimizer.zero_grad()
            outputs = self.model(images) # Output [Batch] (do đã squeeze ở model)
            
            # Tính Loss
            if self.task == "classification":
                loss = self.criterion(outputs, labels.float())
                preds = (torch.sigmoid(outputs) > 0.5).float()
            else:
                loss = self.criterion(outputs, labels.float())
                preds = outputs

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            
            # Lưu lại để tính metric
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            
            # Update thanh loading
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_metric = self.calculate_metric(all_labels, all_preds)
        
        return epoch_loss, epoch_metric

    def validate(self, epoch_index):
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch_index} [VAL]  ")
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                
                if self.task == "classification":
                    loss = self.criterion(outputs, labels.float())
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    loss = self.criterion(outputs, labels.float())
                    preds = outputs

                running_loss += loss.item() * images.size(0)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_metric = self.calculate_metric(all_labels, all_preds)
        
        return epoch_loss, epoch_metric

    def calculate_metric(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if self.task == "classification":
            # Accuracy
            return (y_true == y_pred).mean()
        else:
            # MAE (Sai số tuyệt đối trung bình) cho bài toán cường độ
            return np.mean(np.abs(y_true - y_pred))

    def train(self):
        """
        Hàm chính được gọi từ Main.py
        """
        print(f"🔥 Bắt đầu huấn luyện: Task={self.task} | Device={self.device}")
        
        for epoch in range(1, settings.NUM_EPOCHS + 1):
            train_loss, train_metric = self.train_one_epoch(epoch)
            val_loss, val_metric = self.validate(epoch)

            # In kết quả
            metric_name = "Acc" if self.task == "classification" else "MAE"
            print(f"   Done Epoch {epoch}: Train Loss={train_loss:.4f} {metric_name}={train_metric:.4f} | Val Loss={val_loss:.4f} {metric_name}={val_metric:.4f}")

            # --- CƠ CHẾ LƯU MODEL TỐT NHẤT ---
            save_condition = False
            if self.task == "classification":
                # Classification: Accuracy càng cao càng tốt
                if val_metric > self.best_metric:
                    save_condition = True
                    self.best_metric = val_metric
            else:
                # Regression: MAE càng thấp càng tốt
                if val_metric < self.best_metric:
                    save_condition = True
                    self.best_metric = val_metric

            if save_condition:
                save_path = os.path.join(settings.OUTPUTS_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"   💾 Đã lưu model tốt nhất (Val {metric_name}: {val_metric:.4f})")