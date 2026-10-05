"""
Train LightGBM track model trên IBTrACS Western Pacific.
Đặt trong thư mục train/ cùng các script train khác. Chạy từ đâu cũng được:
    python source/train/train_track.py
"""
import os
import sys
import pickle
import argparse
import urllib.request

import numpy as np
import pandas as pd
import lightgbm as lgb


def find_repo_root():
    """Đi ngược lên từ vị trí file cho tới thư mục chứa 'source/'."""
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(d, "source")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("Không tìm thấy thư mục gốc repo (chứa 'source/')")
        d = parent


ROOT = find_repo_root()
sys.path.insert(0, ROOT)
from source.models.track_model import FEATURES, HORIZONS, build_features, haversine_km  # noqa: E402

IBTRACS_URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
               "v04r01/access/csv/ibtracs.WP.list.v04r01.csv")


def wrap_dlon(d):
    return (d + 180.0) % 360.0 - 180.0


def load_ibtracs(path, min_year):
    if not os.path.exists(path):
        print(f"Downloading IBTrACS WP -> {path}")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(IBTRACS_URL, path)

    cols = ["SID", "SEASON", "ISO_TIME", "LAT", "LON", "WMO_WIND", "USA_WIND"]
    df = pd.read_csv(path, skiprows=[1], usecols=cols, low_memory=False)
    for c in ["SEASON", "LAT", "LON", "WMO_WIND", "USA_WIND"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df["VMAX"] = df["USA_WIND"].fillna(df["WMO_WIND"])

    df = df.dropna(subset=["ISO_TIME", "LAT", "LON", "SEASON"])
    df = df[(df["SEASON"] >= min_year)
            & df["ISO_TIME"].dt.hour.isin([0, 6, 12, 18])
            & (df["ISO_TIME"].dt.minute == 0)]
    df["LON"] = df["LON"] % 360.0
    df = df.drop_duplicates(["SID", "ISO_TIME"]).sort_values(["SID", "ISO_TIME"])
    print(f"Loaded {len(df):,} điểm 6h, {df['SID'].nunique():,} cơn bão (từ {min_year})")
    return df


def make_samples(df):
    rows = []
    H = pd.Timedelta(hours=1)
    for sid, g in df.groupby("SID"):
        g = g.set_index("ISO_TIME")
        if len(g) < 9:
            continue
        idx = set(g.index)
        lat, lon, vmax, season = g["LAT"], g["LON"], g["VMAX"], g["SEASON"]
        for t0 in g.index:
            t48, t24 = t0 - 48 * H, t0 - 24 * H
            futures = [t0 + h * H for h in HORIZONS]
            if t48 not in idx or t24 not in idx or any(t not in idx for t in futures):
                continue
            hist = [(lat[t48], lon[t48]), (lat[t24], lon[t24]), (lat[t0], lon[t0])]
            r = build_features(hist, vmax[t0])
            r["season"] = int(season[t0])
            for h, t in zip(HORIZONS, futures):
                r[f"{h}_dlat"] = lat[t] - lat[t0]
                r[f"{h}_dlon"] = wrap_dlon(lon[t] - lon[t0])
            rows.append(r)
    out = pd.DataFrame(rows)
    print(f"Tạo được {len(out):,} mẫu")
    return out


def train_one(Xtr, ytr, Xva, yva):
    m = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.03, num_leaves=31,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.9, reg_lambda=1.0, verbose=-1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l2",
          callbacks=[lgb.early_stopping(150, verbose=False)])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "data", "ibtracs.WP.list.v04r01.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "track_lgbm.pkl"))
    ap.add_argument("--min_year", type=int, default=1980)
    args = ap.parse_args()

    df = make_samples(load_ibtracs(args.csv, args.min_year))
    tr = df[df["season"] <= 2014]
    va = df[(df["season"] >= 2015) & (df["season"] <= 2016)]
    te = df[df["season"] >= 2017]
    print(f"Train {len(tr):,} | Val {len(va):,} | Test {len(te):,}")

    Xtr, Xva, Xte = (d[FEATURES].values.astype(float) for d in (tr, va, te))
    models, cone_km = {}, {}

    print("\n Horizon | LGBM mean | LGBM p67 | Persist mean | Cải thiện")
    for h in HORIZONS:
        for comp in ("dlat", "dlon"):
            key = f"{h}_{comp}"
            models[key] = train_one(Xtr, tr[key].values, Xva, va[key].values)

        lat0, lon0 = te["lat0"].values, te["lon0"].values
        true_lat = lat0 + te[f"{h}_dlat"].values
        true_lon = lon0 + te[f"{h}_dlon"].values
        p_lat = lat0 + models[f"{h}_dlat"].predict(Xte)
        p_lon = lon0 + models[f"{h}_dlon"].predict(Xte)
        b_lat = lat0 + te["dlat24"].values * h / 24
        b_lon = lon0 + te["dlon24"].values * h / 24

        err = haversine_km(true_lat, true_lon, p_lat, p_lon)
        err_b = haversine_km(true_lat, true_lon, b_lat, b_lon)
        cone_km[h] = float(np.percentile(err, 67))
        gain = (1 - err.mean() / err_b.mean()) * 100
        print(f" {h:>4}h   | {err.mean():8.1f}  | {cone_km[h]:7.1f}  | {err_b.mean():11.1f}  | {gain:5.1f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"models": models, "features": FEATURES,
                     "horizons": HORIZONS, "cone_km": cone_km}, f)
    print(f"\nĐã lưu {args.out}")
    print("Dán vào app.py:  TRACK_CONE_KM = "
          + "{" + ", ".join(f"{h}: {round(v)}" for h, v in cone_km.items()) + "}")


if __name__ == "__main__":
    main()
