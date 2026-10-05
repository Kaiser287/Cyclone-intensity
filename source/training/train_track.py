"""
Train model dự đoán quỹ đạo bão (LightGBM) trên IBTrACS - Tây Bắc Thái Bình Dương.

Chạy (Colab hoặc máy local, CPU là đủ):
    pip install lightgbm scikit-learn joblib pandas
    python train_track.py                       # tự tải IBTrACS WP từ NOAA
    python train_track.py --csv ibtracs.WP.list.v04r01.csv --out models/track_lgbm.pkl

Kết quả:
    models/track_lgbm.pkl          -> bundle joblib cho TrackPredictor
    models/track_metrics.json      -> sai số (km) LightGBM vs Persistence trên tập test

Chia dữ liệu theo năm (tránh rò rỉ giữa các cơn bão):
    train: <= 2015 | val: 2016-2018 (early stopping + nón sai số) | test: >= 2019
"""
import argparse
import json
import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from source.models.track_model import (  # noqa: E402
    FEATURES, HORIZONS, TrackPredictor, build_features, dlon_deg, haversine_km)

IBTRACS_URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
               "climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv")

LGB_PARAMS = dict(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                  min_child_samples=30, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.9, reg_lambda=1.0, verbose=-1)


# ---------------------------------------------------------------- data
def load_ibtracs(src, min_year):
    print(f"[1/4] Đọc IBTrACS: {src}")
    cols = ["SID", "SEASON", "ISO_TIME", "LAT", "LON", "USA_WIND", "WMO_WIND", "TRACK_TYPE"]
    df = pd.read_csv(src, skiprows=[1], usecols=cols, low_memory=False,
                     keep_default_na=False, na_values=["", " "])

    df = df[df["TRACK_TYPE"].astype(str).str.strip() == "main"]
    df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")
    df = df[df["SEASON"] >= min_year]

    df["time"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df["lat"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["lon"] = pd.to_numeric(df["LON"], errors="coerce") % 360.0
    # USA_WIND (JTWC, gió 1 phút) cùng nguồn nhãn với TCIR -> ưu tiên
    usa = pd.to_numeric(df["USA_WIND"], errors="coerce")
    wmo = pd.to_numeric(df["WMO_WIND"], errors="coerce")
    df["vmax"] = usa.fillna(wmo)

    df = df.dropna(subset=["time", "lat", "lon"])
    df = df[(df["time"].dt.minute == 0) & (df["time"].dt.hour % 6 == 0)]  # chỉ mốc synop 6h
    df = df.drop_duplicates(["SID", "time"]).sort_values(["SID", "time"])
    df = df[["SID", "SEASON", "time", "lat", "lon", "vmax"]].reset_index(drop=True)
    print(f"      {df['SID'].nunique()} cơn bão, {len(df)} điểm 6h")
    return df


def _lookup(base, hours):
    """Vị trí của cùng cơn bão tại (time + hours). NaN nếu không có."""
    q = base[["SID", "time"]].copy()
    q["time"] = q["time"] + pd.Timedelta(hours=hours)
    m = q.merge(base[["SID", "time", "lat", "lon"]], on=["SID", "time"], how="left")
    return m["lat"].to_numpy(), m["lon"].to_numpy()


def make_samples(df):
    print("[2/4] Tạo mẫu (history -48h/-24h/0h -> tương lai +12..+48h)")
    s = df.copy()
    s["lat48"], s["lon48"] = _lookup(df, -48)
    s["lat24"], s["lon24"] = _lookup(df, -24)
    for h in HORIZONS:
        lat_h, lon_h = _lookup(df, h)
        s[f"dlat_{h}"] = lat_h - s["lat"].to_numpy()
        s[f"dlon_{h}"] = np.where(np.isnan(lon_h), np.nan, dlon_deg(lon_h, s["lon"].to_numpy()))
    s = s.dropna(subset=["lat48", "lon48", "lat24", "lon24"]).reset_index(drop=True)
    print(f"      {len(s)} mẫu có đủ lịch sử 48h")
    return s


def features_of(s):
    return build_features(s["lat48"], s["lon48"], s["lat24"], s["lon24"],
                          s["lat"], s["lon"], s["vmax"])


def persistence(s, h):
    lat0, lon0 = s["lat"].to_numpy(), s["lon"].to_numpy()
    v_lat = (lat0 - s["lat24"].to_numpy()) / 24.0
    v_lon = dlon_deg(lon0, s["lon24"].to_numpy()) / 24.0
    return lat0 + v_lat * h, lon0 + v_lon * h


def track_error_km(s, h, lat_pred, lon_pred):
    lat_true = s["lat"].to_numpy() + s[f"dlat_{h}"].to_numpy()
    lon_true = s["lon"].to_numpy() + s[f"dlon_{h}"].to_numpy()
    return haversine_km(lat_true, lon_true, lat_pred, lon_pred)


# ---------------------------------------------------------------- train
def train(samples, val_years, test_from):
    print("[3/4] Train LightGBM (mỗi mốc giờ x {dlat, dlon} = 1 regressor)")
    split = np.where(samples["SEASON"] < val_years[0], "train",
                     np.where(samples["SEASON"] < test_from, "val", "test"))
    models, cone_km, metrics = {}, {}, {"val": {}, "test": {}}

    for h in HORIZONS:
        ok = samples[f"dlat_{h}"].notna().to_numpy()
        parts = {k: samples[ok & (split == k)].reset_index(drop=True)
                 for k in ("train", "val", "test")}
        X = {k: features_of(v) for k, v in parts.items()}

        for comp in ("dlat", "dlon"):
            y_tr = parts["train"][f"{comp}_{h}"].to_numpy()
            y_va = parts["val"][f"{comp}_{h}"].to_numpy()
            m = lgb.LGBMRegressor(**LGB_PARAMS)
            m.fit(X["train"], y_tr, eval_set=[(X["val"], y_va)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            models[(h, comp)] = m

        for k in ("val", "test"):
            p, xk = parts[k], X[k]
            if len(p) == 0:
                continue
            lat_p = p["lat"].to_numpy() + models[(h, "dlat")].predict(xk)
            lon_p = p["lon"].to_numpy() + models[(h, "dlon")].predict(xk)
            err_ai = track_error_km(p, h, lat_p, lon_p)
            err_pe = track_error_km(p, h, *persistence(p, h))
            metrics[k][str(h)] = {
                "n": int(len(p)),
                "lgbm_mean_km": round(float(err_ai.mean()), 1),
                "lgbm_median_km": round(float(np.median(err_ai)), 1),
                "persistence_mean_km": round(float(err_pe.mean()), 1),
                "skill_vs_persistence_pct": round(float(100 * (1 - err_ai.mean() / err_pe.mean())), 1),
            }
            if k == "val":
                # Nón sai số kiểu NHC: bán kính chứa 2/3 sai số trên tập val
                cone_km[h] = round(float(np.percentile(err_ai, 66.7)), 0)

        t = metrics["test"].get(str(h), {})
        print(f"      +{h:2d}h | train {len(parts['train']):6d} | "
              f"test LGBM {t.get('lgbm_mean_km', float('nan')):6.1f} km | "
              f"Persistence {t.get('persistence_mean_km', float('nan')):6.1f} km | "
              f"skill {t.get('skill_vs_persistence_pct', float('nan')):5.1f}% | cone {cone_km[h]:.0f} km")
    return models, cone_km, metrics


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=IBTRACS_URL, help="đường dẫn hoặc URL file IBTrACS WP")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "models", "track_lgbm.pkl"))
    ap.add_argument("--min-year", type=int, default=1980, help="bỏ dữ liệu trước vệ tinh hiện đại")
    ap.add_argument("--val-start", type=int, default=2016)
    ap.add_argument("--test-start", type=int, default=2019)
    args = ap.parse_args()

    df = load_ibtracs(args.csv, args.min_year)
    samples = make_samples(df)
    models, cone_km, metrics = train(samples, (args.val_start,), args.test_start)
    metrics["split"] = {"train": f"{args.min_year}-{args.val_start - 1}",
                        "val": f"{args.val_start}-{args.test_start - 1}",
                        "test": f">={args.test_start}"}

    print("[4/4] Lưu model")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump({"models": models, "horizons": HORIZONS, "features": FEATURES,
                 "cone_km": cone_km, "metrics": metrics,
                 "lightgbm_version": lgb.__version__}, args.out)
    metrics_path = os.path.join(os.path.dirname(args.out) or ".", "track_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Kiểm tra: load lại đúng như app sẽ load
    tp = TrackPredictor(args.out)
    assert tp.is_ai, f"Load lại thất bại: {tp.load_error}"
    demo = tp.predict([(13.0, 116.0), (14.5, 114.0), (16.0, 112.0)], vmax=80)
    print(f"      Đã lưu {args.out} + {metrics_path}")
    print(f"      Demo: {[(p['hour'], p['lat'], p['lon']) for p in demo['points']]}")


if __name__ == "__main__":
    main()
