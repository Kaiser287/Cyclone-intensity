"""
Cyclone Intensity & Track - Streamlit app

- Cường độ : ResNet18 (1 kênh IR) train trên TCIR WPAC  -> predictor.predict()
- Quỹ đạo  : LightGBM train trên IBTrACS WPAC            -> predictor.track.predict()
             (tự fallback Persistence nếu chưa có models/track_lgbm.pkl)

Chạy: streamlit run app.py
"""
import io
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from source.inference.predictor import CATEGORIES, IntensityPredictor  # noqa: E402

CKPT_CANDIDATES = [
    os.path.join(ROOT, "source", "models", "intensity_best.pt"),
    os.path.join(ROOT, "models", "intensity_best.pt"),
    os.path.join(ROOT, "outputs", "intensity_best.pt"),
    os.path.join(ROOT, "intensity_best.pt"),
]

st.set_page_config(page_title="Cyclone Intensity & Track", page_icon="🌀", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Đang tải model...")
def load_predictor():
    ckpt = next((p for p in CKPT_CANDIDATES if os.path.exists(p)), None)
    if ckpt is None:
        raise FileNotFoundError(
            "Không tìm thấy checkpoint cường độ. Đặt file intensity_best.pt vào thư mục models/.")
    return IntensityPredictor(ckpt)


def read_upload(uploaded):
    """File upload -> object mà predictor.to_bt() hiểu (np.ndarray hoặc PIL.Image)."""
    data = uploaded.getvalue()
    if uploaded.name.lower().endswith(".npy"):
        return np.load(io.BytesIO(data), allow_pickle=False)
    return Image.open(io.BytesIO(data))


def cone_polygon(lat, lon, radius_km, n=48):
    """Vòng tròn bán kính radius_km quanh (lat, lon), trả về list lat, list lon."""
    t = np.linspace(0, 2 * np.pi, n)
    dlat = radius_km / 111.2 * np.cos(t)
    dlon = radius_km / (111.2 * max(np.cos(np.radians(lat)), 0.1)) * np.sin(t)
    return list(lat + dlat), list(lon + dlon)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🌀 Cyclone AI")
uploaded = st.sidebar.file_uploader("Ảnh vệ tinh IR (.npy Kelvin hoặc PNG/JPG)",
                                    type=["npy", "png", "jpg", "jpeg"])
use_tta = st.sidebar.checkbox("Test-time augmentation (xoay 4 hướng)", value=True)

st.sidebar.subheader("Vị trí tâm bão")
c1, c2 = st.sidebar.columns(2)
lat0 = c1.number_input("Lat hiện tại", -40.0, 60.0, 15.0, 0.1)
lon0 = c2.number_input("Lon hiện tại", 90.0, 200.0, 130.0, 0.1)
lat24 = c1.number_input("Lat -24h", -40.0, 60.0, 13.5, 0.1)
lon24 = c2.number_input("Lon -24h", 90.0, 200.0, 133.0, 0.1)
lat48 = c1.number_input("Lat -48h", -40.0, 60.0, 12.0, 0.1)
lon48 = c2.number_input("Lon -48h", 90.0, 200.0, 136.0, 0.1)
history = [(lat48, lon48), (lat24, lon24), (lat0, lon0)]

# ---------------------------------------------------------------- model
try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Lỗi tải model cường độ: {e}")
    st.stop()

with st.sidebar.expander("Thông tin model", expanded=False):
    st.write(f"**Cường độ:** ResNet18 IR · device `{predictor.device}`")
    if predictor.track.is_ai:
        st.write(f"**Quỹ đạo:** {predictor.track.method} (AI) ✅")
        if predictor.track.metrics:
            st.json(predictor.track.metrics)
    else:
        st.write(f"**Quỹ đạo:** {predictor.track.method} (baseline, không phải AI)")
        if predictor.track.load_error:
            st.warning(predictor.track.load_error)

# ---------------------------------------------------------------- main
st.title("Ước lượng cường độ & dự báo quỹ đạo bão")

if uploaded is None:
    st.info("Tải lên ảnh IR ở thanh bên trái để bắt đầu. "
            "File .npy (nhiệt độ sáng Kelvin, như TCIR) cho kết quả chính xác nhất.")
    st.stop()

try:
    raw = read_upload(uploaded)
    bt, approx = predictor.to_bt(raw)
    res = predictor.predict(bt.copy(), tta=use_tta)
    res["approx_input"] = approx
    track = predictor.track.predict(history, res["wind_speed"])
except Exception as e:
    st.error(f"Lỗi khi dự đoán: {type(e).__name__}: {e}")
    st.stop()

if approx:
    st.warning("Ảnh PNG/JPG được quy đổi pixel → Kelvin theo công thức xấp xỉ, "
               "kết quả chỉ mang tính tham khảo. Dùng .npy để chính xác hơn.")

tab_metrics, tab_3d, tab_track, tab_report = st.tabs(["📊 METRICS", "🧊 3D VIEW", "🗺️ TRACK", "📄 REPORT"])

# ---------------------------------------------------------------- METRICS
with tab_metrics:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sức gió (kt)", f"{res['wind_speed']}")
    m2.metric("Sức gió (km/h)", f"{res['wind_kmh']}")
    m3.metric("Cấp", res["category"])
    m4.metric("BT lạnh nhất (K)", f"{np.nanmin(bt):.1f}")

    st.markdown(
        f"<div style='padding:12px;border-radius:8px;background:{res['color']};"
        f"color:#111;font-weight:600;text-align:center'>{res['lifecycle']}</div>",
        unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.image(predictor.bt_to_display(bt), caption="Ảnh IR (trắng = mây lạnh)",
                 use_container_width=True)
    with right:
        st.subheader("Thang Saffir-Simpson")
        lower = 0
        rows = []
        for upper, code, label, _ in CATEGORIES:
            rows.append({"Cấp": code, "Tên": label,
                         "Khoảng (kt)": f"{lower}–{upper - 1}" if upper < 999 else f"≥ {lower}",
                         "Hiện tại": "◀" if code == res["category"] else ""})
            lower = upper
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- 3D VIEW
with tab_3d:
    step = max(1, max(bt.shape) // 100)
    z = np.nan_to_num(bt[::step, ::step], nan=float(np.nanmax(bt)))
    fig3d = go.Figure(go.Surface(z=-z, surfacecolor=z, colorscale="Turbo_r",
                                 colorbar=dict(title="BT (K)")))
    fig3d.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0),
                        scene=dict(zaxis=dict(title="− BT (đỉnh mây cao hơn)"),
                                   xaxis_title="x", yaxis_title="y"),
                        title="Cấu trúc đỉnh mây (mây càng lạnh càng cao)")
    st.plotly_chart(fig3d, use_container_width=True)

# ---------------------------------------------------------------- TRACK
with tab_track:
    if track["is_ai"]:
        st.success(f"Dự báo bằng **{track['method']}** train trên IBTrACS (AI).")
    else:
        st.warning(f"Đang dùng **{track['method']}** (ngoại suy tuyến tính). "
                   "Chạy train_track.py để tạo models/track_lgbm.pkl và bật LightGBM.")

    hist = track["history"]
    pts = track["points"]
    fig = go.Figure()
    for p in pts:  # nón sai số
        la, lo = cone_polygon(p["lat"], p["lon"], p["cone_km"])
        fig.add_trace(go.Scattergeo(lat=la, lon=lo, mode="lines", fill="toself",
                                    fillcolor="rgba(255,152,0,0.12)",
                                    line=dict(color="rgba(255,152,0,0.5)", width=1),
                                    hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattergeo(lat=[h["lat"] for h in hist], lon=[h["lon"] for h in hist],
                                mode="lines+markers", name="Quá khứ",
                                line=dict(color="#90A4AE", width=2),
                                text=[f"{h['hour']}h" for h in hist]))
    fig.add_trace(go.Scattergeo(lat=[track["origin"]["lat"]] + [p["lat"] for p in pts],
                                lon=[track["origin"]["lon"]] + [p["lon"] for p in pts],
                                mode="lines+markers+text", name=f"Dự báo ({track['method']})",
                                line=dict(color=res["color"], width=3),
                                text=["Now"] + [f"+{p['hour']}h" for p in pts],
                                textposition="top right"))
    all_lat = [h["lat"] for h in hist] + [p["lat"] for p in pts]
    all_lon = [h["lon"] for h in hist] + [p["lon"] for p in pts]
    fig.update_geos(projection_type="mercator", showland=True, landcolor="#2b2b2b",
                    showocean=True, oceancolor="#0d1b2a", showcountries=True,
                    countrycolor="#555", coastlinecolor="#888",
                    lataxis_range=[min(all_lat) - 6, max(all_lat) + 6],
                    lonaxis_range=[min(all_lon) - 8, max(all_lon) + 8])
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pd.DataFrame([{"Mốc": f"+{p['hour']}h", "Lat": p["lat"], "Lon": p["lon"],
                                "Bán kính nón (km)": p["cone_km"]} for p in pts]),
                 hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- REPORT
with tab_report:
    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "input_file": uploaded.name,
        "intensity": {k: res[k] for k in ("wind_speed", "wind_kmh", "category", "lifecycle",
                                          "approx_input")},
        "track": {"method": track["method"], "is_ai": track["is_ai"],
                  "history": track["history"], "forecast": track["points"]},
    }
    lines = [
        "BÁO CÁO PHÂN TÍCH BÃO",
        f"Thời gian      : {report['generated_at']}",
        f"File đầu vào   : {uploaded.name}" + (" (quy đổi xấp xỉ)" if approx else ""),
        "",
        f"Sức gió ước tính: {res['wind_speed']} kt ({res['wind_kmh']} km/h)",
        f"Phân cấp        : {res['category']} - {res['lifecycle']}",
        "",
        f"Dự báo quỹ đạo ({track['method']}{', AI' if track['is_ai'] else ', baseline'}):",
        f"  Hiện tại: {track['origin']['lat']:.2f}N, {track['origin']['lon']:.2f}E",
    ]
    lines += [f"  +{p['hour']:>2}h  : {p['lat']:.2f}N, {p['lon']:.2f}E  (±{p['cone_km']:.0f} km)"
              for p in pts]
    text = "\n".join(lines)
    st.code(text, language=None)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Tải báo cáo .txt", text, file_name="cyclone_report.txt")
    d2.download_button("⬇️ Tải báo cáo .json", json.dumps(report, ensure_ascii=False, indent=2),
                       file_name="cyclone_report.json", mime="application/json")
