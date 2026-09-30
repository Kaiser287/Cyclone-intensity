"""
Tiền xử lý TCIR (HDF5) cho intensity model.

Việc file này làm:
  1. Đọc `info` (nhãn) và `matrix` (N x 201 x 201 x 4) theo lô, không nạp hết vào RAM.
  2. Lọc theo vùng biển (mặc định WPAC).
  3. Chia train/val theo NĂM CỦA CƠN BÃO (không chia theo frame -> không rò rỉ dữ liệu).
  4. Bỏ frame có quá nhiều NaN, crop tâm ảnh về 128x128, chọn kênh.
  5. Chuẩn hoá bằng mean/std tính từ TẬP TRAIN, rồi thay NaN = 0.
  6. Ghi ra {split}_X.npy (float16) + {split}_meta.csv + stats.json.

Chạy trên Colab:
  python source/data/preprocessor.py \
      --main_h5 /content/tcir/TCIR-ATLN_EPAC_WPAC.h5 \
      --test_h5 /content/tcir/TCIR-ALL_2017.h5 \
      --out_dir /content/processed --channels IR1
"""
import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd

REQUIRED_COLS = ["data_set", "ID", "time", "Vmax", "lat", "lon"]
OPTIONAL_COLS = ["MSLP", "R35_4qAVG"]
CHANNEL_NAMES = ["IR1", "WV", "VIS", "PMW"]


# ---------------------------------------------------------------- nhãn
def load_info(h5_path):
    info = pd.read_hdf(h5_path, key="info", mode="r").reset_index(drop=True)
    missing = [c for c in REQUIRED_COLS if c not in info.columns]
    if missing:
        raise KeyError(f"{h5_path}: thiếu cột {missing}. Cột hiện có: {info.columns.tolist()}")
    info["year"] = info["time"].astype(str).str[:4].astype(int)
    if not info["year"].between(1990, 2030).all():
        raise ValueError(f"Định dạng cột 'time' không như dự kiến, ví dụ: {info['time'].head().tolist()}")
    # Năm của cơn bão = năm của frame đầu tiên -> bão vắt qua năm mới vẫn nằm trọn 1 tập
    info["storm_year"] = info.groupby(["data_set", "ID"])["year"].transform("min")
    return info


def basin_mask(info, basin):
    m = (info["data_set"] == basin).values
    if not m.any():
        raise ValueError(f"Không có frame nào thuộc '{basin}'. Giá trị có trong data_set: "
                         f"{info['data_set'].unique().tolist()}")
    return m


# ---------------------------------------------------------------- ảnh
def read_chunk(X, start, end, channels, crop):
    h = X.shape[1]
    s = (h - crop) // 2
    x = X[start:end, s:s + crop, s:s + crop, :]      # crop ngay khi đọc từ đĩa
    return x[..., channels].astype(np.float32)


def scan(h5_path, keep_mask, channels, crop, max_nan_frac, stats_mask=None, chunk=512):
    """Lượt 1: đánh dấu frame hợp lệ + (tuỳ chọn) tính mean/std trên stats_mask."""
    n = len(keep_mask)
    valid = np.zeros(n, dtype=bool)
    C = len(channels)
    s1, s2, cnt = np.zeros(C), np.zeros(C), np.zeros(C)
    with h5py.File(h5_path, "r") as f:
        X = f["matrix"]
        assert X.shape[0] == n, "Số frame trong matrix và info không khớp"
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            km = keep_mask[start:end]
            if not km.any():
                continue
            x = read_chunk(X, start, end, channels, crop)
            nan_frac = np.isnan(x).mean(axis=(1, 2, 3))
            ok = km & (nan_frac <= max_nan_frac)
            valid[start:end] = ok
            if stats_mask is not None:
                sel = ok & stats_mask[start:end]
                if sel.any():
                    xs = x[sel].astype(np.float64)
                    s1 += np.nansum(xs, axis=(0, 1, 2))
                    s2 += np.nansum(xs ** 2, axis=(0, 1, 2))
                    cnt += np.sum(~np.isnan(xs), axis=(0, 1, 2))
            print(f"\r  scan {end}/{n}", end="")
    print()
    if stats_mask is None:
        return valid, None, None
    mean = s1 / cnt
    std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 1e-12))
    return valid, mean, std


def write_split(h5_path, info, mask, channels, crop, mean, std, out_prefix, chunk=512):
    """Lượt 2: chuẩn hoá và ghi ra .npy (memmap, không tốn RAM)."""
    idx = np.where(mask)[0]
    X_out = np.lib.format.open_memmap(out_prefix + "_X.npy", mode="w+", dtype=np.float16,
                                      shape=(len(idx), crop, crop, len(channels)))
    pos = 0
    with h5py.File(h5_path, "r") as f:
        X = f["matrix"]
        for start in range(0, len(mask), chunk):
            end = min(start + chunk, len(mask))
            m = mask[start:end]
            if not m.any():
                continue
            x = read_chunk(X, start, end, channels, crop)[m]
            x = (x - mean) / std
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = np.clip(x, -10, 10)                      # an toàn cho float16
            X_out[pos:pos + len(x)] = x.astype(np.float16)
            pos += len(x)
    X_out.flush()
    del X_out
    cols = ["ID", "data_set", "time", "year", "storm_year", "Vmax", "lat", "lon"]
    cols += [c for c in OPTIONAL_COLS if c in info.columns]
    info.loc[idx, cols].to_csv(out_prefix + "_meta.csv", index=False)
    return len(idx)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main_h5", required=True, help="TCIR-ATLN_EPAC_WPAC.h5 (2003-2016)")
    p.add_argument("--test_h5", default=None, help="TCIR-ALL_2017.h5 (tập test độc lập)")
    p.add_argument("--out_dir", default="/content/processed")
    p.add_argument("--basin", default="WPAC")
    p.add_argument("--channels", default="IR1", help="VD: IR1 hoặc IR1,PMW")
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--train_end", type=int, default=2014, help="storm_year <= giá trị này -> train")
    p.add_argument("--max_nan_frac", type=float, default=0.1)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    channels = sorted(CHANNEL_NAMES.index(c.strip()) for c in args.channels.split(","))
    ch_names = [CHANNEL_NAMES[c] for c in channels]
    print("Kênh:", ch_names)

    info = load_info(args.main_h5)
    in_basin = basin_mask(info, args.basin)
    train_m = in_basin & (info["storm_year"] <= args.train_end).values
    val_m = in_basin & (info["storm_year"] > args.train_end).values

    print("Lượt 1: kiểm tra NaN + tính mean/std trên train")
    valid, mean, std = scan(args.main_h5, in_basin, channels, args.crop,
                            args.max_nan_frac, stats_mask=train_m)
    train_m &= valid
    val_m &= valid

    ids_tr = set(info.loc[train_m, "ID"])
    ids_va = set(info.loc[val_m, "ID"])
    assert not (ids_tr & ids_va), "Có cơn bão nằm ở cả train và val!"

    print("Lượt 2: ghi file")
    counts = {
        "train": write_split(args.main_h5, info, train_m, channels, args.crop, mean, std,
                             os.path.join(args.out_dir, "train")),
        "val": write_split(args.main_h5, info, val_m, channels, args.crop, mean, std,
                           os.path.join(args.out_dir, "val")),
    }

    if args.test_h5:
        info_t = load_info(args.test_h5)
        test_m = basin_mask(info_t, args.basin)
        valid_t, _, _ = scan(args.test_h5, test_m, channels, args.crop, args.max_nan_frac)
        counts["test"] = write_split(args.test_h5, info_t, test_m & valid_t, channels, args.crop,
                                     mean, std, os.path.join(args.out_dir, "test"))

    vmax_tr = info.loc[train_m, "Vmax"]
    stats = {
        "channels": ch_names,
        "crop": args.crop,
        "basin": args.basin,
        "train_end": args.train_end,
        "img_mean": mean.tolist(),
        "img_std": std.tolist(),
        "vmax_mean": float(vmax_tr.mean()),   # dùng để chuẩn hoá nhãn khi train
        "vmax_std": float(vmax_tr.std()),     # và ĐẢO NGƯỢC khi suy luận
        "counts": counts,
        "storms": {"train": len(ids_tr), "val": len(ids_va)},
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
