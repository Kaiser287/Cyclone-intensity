"""
train_intensity.py — Huấn luyện mô hình CNN ước lượng cường độ bão (Vmax) từ ảnh IR1.
Dữ liệu: output của preprocess (train/val/test_X.npy + *_meta.csv + stats.json).

Chạy trên Colab:
  !python source/train_intensity.py \
      --data_dir /content/drive/MyDrive/cyclone_v2/processed \
      --out_dir  /content/drive/MyDrive/cyclone_v2/models
"""
import os, json, time, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision


# ----------------------------- Tham số -----------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/content/drive/MyDrive/cyclone_v2/processed")
    p.add_argument("--out_dir", default="/content/drive/MyDrive/cyclone_v2/models")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--pretrained", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    args, _ = p.parse_known_args()
    return args


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------- Dữ liệu -----------------------------
def find_vmax_col(df):
    for c in ["Vmax", "vmax", "VMAX", "intensity"]:
        if c in df.columns:
            return c
    raise KeyError(f"Không tìm thấy cột Vmax trong meta. Các cột hiện có: {list(df.columns)}")


def load_split(data_dir, split, stats):
    X = np.load(os.path.join(data_dir, f"{split}_X.npy")).astype(np.float32)
    meta = pd.read_csv(os.path.join(data_dir, f"{split}_meta.csv"))
    # Đưa về dạng (N, C, H, W)
    if X.ndim == 3:
        X = X[:, None]
    elif X.shape[-1] in (1, 2, 3, 4) and X.shape[1] not in (1, 2, 3, 4):
        X = np.transpose(X, (0, 3, 1, 2))
    # Chuẩn hóa ảnh nếu dữ liệu vẫn còn ở đơn vị Kelvin
    if np.nanmean(X[:200]) > 50:
        mean = np.array(stats["img_mean"], dtype=np.float32).reshape(1, -1, 1, 1)
        std = np.array(stats["img_std"], dtype=np.float32).reshape(1, -1, 1, 1)
        X = (X - mean) / std
    X = np.nan_to_num(X, nan=0.0)
    y = meta[find_vmax_col(meta)].values.astype(np.float32)
    y_norm = (y - stats["vmax_mean"]) / stats["vmax_std"]
    print(f"[{split}] X={X.shape}  Vmax: min={y.min():.0f} max={y.max():.0f} mean={y.mean():.1f}")
    return X, y_norm.astype(np.float32), meta


class TCDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i])
        if self.augment:
            # Bão gần như bất biến khi xoay/lật -> tăng cường dữ liệu an toàn
            x = torch.rot90(x, random.randint(0, 3), dims=(1, 2))
            if random.random() < 0.5:
                x = torch.flip(x, dims=(2,))
            # Dịch tâm nhẹ (±4 px) để mô hình chịu được sai số định tâm
            dx, dy = random.randint(-4, 4), random.randint(-4, 4)
            x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
        return x, torch.tensor(self.y[i])


# ----------------------------- Mô hình -----------------------------
def build_model(in_ch, pretrained=True):
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    old = m.conv1
    m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        with torch.no_grad():  # Gộp trọng số RGB thành kênh IR
            w = old.weight.mean(dim=1, keepdim=True)
            m.conv1.weight.copy_(w.repeat(1, in_ch, 1, 1))
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 1))
    return m


# ----------------------------- Huấn luyện / đánh giá -----------------------------
@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval()
    preds = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            if tta:  # Trung bình 4 góc xoay
                p = sum(model(torch.rot90(x, k, dims=(2, 3))).float() for k in range(4)) / 4
            else:
                p = model(x).float()
        preds.append(p.squeeze(1).cpu().numpy())
    return np.concatenate(preds)


def metrics(pred_kt, true_kt):
    err = pred_kt - true_kt
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "Bias": float(np.mean(err)),
    }


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Thiết bị:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    if device.type != "cuda":
        print("⚠️  Không có GPU — hãy đổi Runtime sang T4 GPU.")

    with open(os.path.join(args.data_dir, "stats.json")) as f:
        stats = json.load(f)
    vm, vs = stats["vmax_mean"], stats["vmax_std"]

    Xtr, ytr, _ = load_split(args.data_dir, "train", stats)
    Xva, yva, _ = load_split(args.data_dir, "val", stats)
    Xte, yte, meta_te = load_split(args.data_dir, "test", stats)

    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    tr_loader = DataLoader(TCDataset(Xtr, ytr, augment=True), shuffle=True, drop_last=True, **kw)
    va_loader = DataLoader(TCDataset(Xva, yva), shuffle=False, **kw)
    te_loader = DataLoader(TCDataset(Xte, yte), shuffle=False, **kw)

    model = build_model(Xtr.shape[1], bool(args.pretrained)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(tr_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    best_rmse, bad, history = 1e9, 0, []
    ckpt = os.path.join(args.out_dir, "intensity_best.pt")

    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for x, y in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x).squeeze(1).float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item() * x.size(0)

        pv = predict(model, va_loader, device) * vs + vm
        m = metrics(pv, yva * vs + vm)
        history.append({"epoch": ep, "train_loss": run / len(tr_loader.dataset), **m})
        print(f"Epoch {ep:02d} | loss {history[-1]['train_loss']:.4f} | "
              f"val RMSE {m['RMSE']:.2f} kt | MAE {m['MAE']:.2f} | bias {m['Bias']:+.2f} | "
              f"{time.time() - t0:.0f}s")

        if m["RMSE"] < best_rmse:
            best_rmse, bad = m["RMSE"], 0
            torch.save({"model": model.state_dict(), "stats": stats,
                        "in_ch": Xtr.shape[1], "epoch": ep}, ckpt)
            print(f"   ✅ Lưu mô hình tốt nhất (val RMSE {best_rmse:.2f} kt)")
        else:
            bad += 1
            # Chỉ cho dừng sớm ở nửa sau, khi learning rate đã giảm (OneCycle)
            if bad >= args.patience and ep > args.epochs // 2:
                print(f"Dừng sớm sau {ep} epoch.")
                break

    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "intensity_history.csv"), index=False)

    # ----------------------------- Đánh giá trên tập test 2017 -----------------------------
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    true_te = yte * vs + vm
    p_plain = predict(model, te_loader, device) * vs + vm
    p_tta = predict(model, te_loader, device, tta=True) * vs + vm
    m_plain, m_tta = metrics(p_plain, true_te), metrics(p_tta, true_te)

    print("\n===== KẾT QUẢ TEST (WPAC 2017) =====")
    print(f"Không TTA : RMSE {m_plain['RMSE']:.2f} kt | MAE {m_plain['MAE']:.2f} | bias {m_plain['Bias']:+.2f}")
    print(f"Có TTA    : RMSE {m_tta['RMSE']:.2f} kt | MAE {m_tta['MAE']:.2f} | bias {m_tta['Bias']:+.2f}")

    # Sai số theo cấp bão (Saffir-Simpson, knots)
    bins = [0, 34, 64, 83, 96, 113, 137, 250]
    names = ["TD", "TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5"]
    cat = pd.cut(true_te, bins=bins, labels=names, right=False)
    df = pd.DataFrame({"true": true_te, "pred": p_tta, "cat": cat})
    df["abs_err"] = (df.pred - df.true).abs()
    print("\nMAE theo cấp bão:")
    print(df.groupby("cat", observed=True)["abs_err"].agg(["count", "mean"]).round(2))

    out = meta_te.copy()
    out["Vmax_pred"] = p_tta
    out.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)
    with open(os.path.join(args.out_dir, "intensity_metrics.json"), "w") as f:
        json.dump({"best_val_RMSE": best_rmse, "test_plain": m_plain, "test_tta": m_tta,
                   "best_epoch": state["epoch"]}, f, indent=2)
    print(f"\nĐã lưu mô hình, lịch sử và dự đoán vào: {args.out_dir}")


if __name__ == "__main__":
    main()
