import os
import sys

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image

# --- SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from source.inference.predictor import IntensityPredictor
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()

# Đường dẫn checkpoint: thử lần lượt các vị trí
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "models", "intensity_best.pt"),
    os.path.join(BASE_DIR, "outputs", "intensity_best.pt"),
]

# Kết quả đánh giá thật của model (từ intensity_metrics.json)
MODEL_INFO = {
    "arch": "ResNet18 (1-channel IR1, 128x128)",
    "data": "TCIR – Western North Pacific",
    "val_rmse": 11.58,
    "test_rmse": 10.10,
    "test_year": 2017,
}

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Typhoon Analytics Core", page_icon="🌪️", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1, h2, h3 { color: #ffffff !important; }
    .stButton>button { background-color: #2962ff; color: white; font-weight: bold; border: none; height: 50px; width: 100%; border-radius: 8px; }
    div[data-testid="stMetricValue"] { color: #00e676 !important; font-size: 26px !important; }
    </style>
    """, unsafe_allow_html=True)


# --- LOAD MODEL ---
@st.cache_resource
def load_ai_engine():
    path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        return None, "Model file not found (models/intensity_best.pt)"
    try:
        return IntensityPredictor(path), None
    except Exception as e:
        return None, str(e)


# --- INPUT HANDLING ---
def read_upload(uploaded_file, predictor):
    """Trả về (input_cho_model, ảnh_PIL_để_hiển_thị)."""
    if uploaded_file.name.lower().endswith(".npy"):
        arr = np.load(uploaded_file)
        bt, _ = predictor.to_bt(arr)
        return arr, predictor.bt_to_display(bt)
    img = Image.open(uploaded_file).convert("L")
    return img, img


# --- 1. 3D CLOUD-TOP VISUALIZATION ---
def generate_expanded_3d_data(image, center_lat, center_lon):
    pixels = np.array(image.resize((80, 80)).convert("L"))
    lat_step = lon_step = 0.08
    rows, cols = pixels.shape
    data = []
    for r in range(rows):
        for c in range(cols):
            b = int(pixels[r, c])
            if b <= 30:
                continue
            if b < 100:
                color = [80, 80, 80, 180]
            elif b < 180:
                color = [200, 200, 200, 200]
            else:
                color = [b, 50, 50, 255]
            data.append({
                "lat": float(center_lat - (r - rows / 2) * lat_step),
                "lon": float(center_lon + (c - cols / 2) * lon_step),
                "height": float(b / 255.0 * 90000),
                "color": color,
            })
    return pd.DataFrame(data)


# --- 2. TRACK EXTRAPOLATION (RULE-BASED, KHÔNG PHẢI ML) ---
def predict_track_cliper(lat_now, lon_now, lat_24, lon_24, lat_48, lon_48, wind_kt):
    """Ngoại suy kiểu CLIPER: vận tốc quá khứ + dòng dẫn theo vĩ độ + beta drift.
    Cường độ (từ model CNN) chỉ dùng để chỉnh quán tính và beta drift."""
    v_lat = 0.6 * (lat_now - lat_24) / 4.0 + 0.4 * (lat_24 - lat_48) / 4.0
    v_lon = 0.6 * (lon_now - lon_24) / 4.0 + 0.4 * (lon_24 - lon_48) / 4.0

    f = min(wind_kt / 150.0, 1.0)
    inertia = 0.7 + f * 0.25
    beta_lat, beta_lon = 0.05 + f * 0.05, -0.05

    v_lat, v_lon = v_lat * 1.8, v_lon * 1.8
    speed = (v_lat ** 2 + v_lon ** 2) ** 0.5
    if speed < 0.5:
        s = 0.5 / speed if speed > 0 else 1.0
        v_lat, v_lon = v_lat * s, v_lon * s

    track = [
        {"lat": lat_48, "lon": lon_48, "type": "History (-48h)", "size": 3000, "color": [100, 100, 100]},
        {"lat": lat_24, "lon": lon_24, "type": "History (-24h)", "size": 4000, "color": [150, 150, 150]},
        {"lat": lat_now, "lon": lon_now, "type": "CURRENT CENTER", "size": 12000, "color": [255, 0, 0]},
    ]
    cone = []
    lat, lon, radius = lat_now, lon_now, 50

    for step in range(1, 9):  # 8 bước x 6h = 48h
        if lat < 20.0:
            s_lon, s_lat = -0.35, 0.08
        elif lat < 25.0:
            s_lon, s_lat = -0.15 + (lat - 20.0) * 0.05, 0.2
        else:
            s_lon, s_lat = 0.4 + (lat - 25.0) * 0.15, 0.25

        v_lat = (v_lat * inertia + (s_lat + beta_lat) * (1 - inertia)) * 1.05
        v_lon = (v_lon * inertia + (s_lon + beta_lon) * (1 - inertia)) * 1.05
        lat, lon = lat + v_lat, lon + v_lon
        radius += 20

        hour = step * 6
        if hour in (24, 36, 48):
            track.append({"lat": float(lat), "lon": float(lon), "type": f"Forecast (+{hour}h)",
                          "size": 6000 + step * 300, "color": [255, 165, 0]})
            cone.append({"lat": float(lat), "lon": float(lon), "radius": radius * 1000})

    return pd.DataFrame(track), cone


# ================= MAIN INTERFACE =================
st.title("🌪️ AI TYPHOON ANALYTICS CORE")
st.caption("CNN Intensity Estimation (TCIR) | 3D Cloud-Top View | CLIPER-style Track Extrapolation")
st.markdown("---")

for k in ("result", "img_cache", "file_id"):
    st.session_state.setdefault(k, None)

predictor, err = load_ai_engine()

# === SIDEBAR ===
with st.sidebar:
    st.header("⚙️ Control Panel")
    uploaded_file = st.file_uploader("Upload IR image (.npy Kelvin preferred, or PNG/JPG):",
                                     type=["npy", "png", "jpg", "jpeg"])
    st.markdown("---")
    st.subheader("🌐 Coordinates Input")
    lat_input = st.number_input("Current Latitude (Now)", value=16.0)
    lon_input = st.number_input("Current Longitude (Now)", value=112.0)
    st.markdown("🔻 **History Data**")
    lat_24 = st.number_input("Lat (-24h ago)", value=14.5)
    lon_24 = st.number_input("Lon (-24h ago)", value=114.0)
    lat_48 = st.number_input("Lat (-48h ago)", value=13.0)
    lon_48 = st.number_input("Lon (-48h ago)", value=116.0)
    st.markdown("---")
    st.caption(f"Model: {MODEL_INFO['arch']}\n\n"
               f"Test RMSE ({MODEL_INFO['test_year']}): {MODEL_INFO['test_rmse']} kt")

# Đổi file thì xoá kết quả cũ
if uploaded_file is not None and uploaded_file.file_id != st.session_state["file_id"]:
    st.session_state.update(result=None, img_cache=None, file_id=uploaded_file.file_id)

# === MAIN LAYOUT ===
col_left, col_right = st.columns([1, 1.8])

with col_left:
    st.subheader("📡 Input Imagery")
    if err:
        st.error(err)
    elif uploaded_file is None:
        st.info("Waiting for data stream...")
    else:
        try:
            model_input, display_img = read_upload(uploaded_file, predictor)
            st.image(display_img, caption="Infrared (white = cold cloud tops)", use_container_width=True)
            if st.button("🚀 EXECUTE ANALYSIS"):
                with st.spinner("Running CNN inference (4x TTA)..."):
                    st.session_state["result"] = predictor.predict(model_input)
                    st.session_state["img_cache"] = display_img
        except Exception as e:
            st.error(f"Cannot read file: {e}")

with col_right:
    res = st.session_state["result"]
    if res:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 METRICS", "🧊 3D VIEW", "🗺️ TRACK", "📜 REPORT"])

        with tab1:
            c1, c2 = st.columns(2)
            c1.metric("Max Sustained Wind", f"{res['wind_kmh']} km/h")
            c2.metric("Wind Speed (Knots)", f"{res['wind_speed']} kt",
                      delta=f"± ~{MODEL_INFO['test_rmse']:.0f} kt (test RMSE)", delta_color="off")
            st.markdown(f"<div style='color:{res['color']}; font-weight:bold; font-size:22px; margin-top:10px'>"
                        f"{res['category']} – {res['lifecycle']}</div>", unsafe_allow_html=True)
            st.progress(min(res["wind_speed"] / 185, 1.0))
            if res["approx_input"]:
                st.warning("PNG/JPG input: pixel values are converted to Kelvin approximately. "
                           "Use a TCIR .npy file for accurate results.")

        with tab2:
            st.info("💡 Height = cloud-top coldness (visual only). Right-click to rotate, scroll to zoom.")
            df_3d = generate_expanded_3d_data(st.session_state["img_cache"], lat_input, lon_input)
            layer = pdk.Layer("ColumnLayer", data=df_3d, get_position=["lon", "lat"],
                              get_elevation="height", elevation_scale=2, radius=6000,
                              get_fill_color="color", pickable=True, auto_highlight=True)
            view = pdk.ViewState(latitude=float(lat_input), longitude=float(lon_input), zoom=6, pitch=55)
            st.pydeck_chart(pdk.Deck(initial_view_state=view, layers=[layer], map_style="dark"))

        with tab3:
            st.write("Rule-based CLIPER-style extrapolation (not a trained model).")
            st.info(f"Inertia and beta drift are scaled by CNN intensity: **{res['wind_speed']} kt**")
            if st.button("📍 GENERATE FORECAST TRACK"):
                track_df, cone = predict_track_cliper(lat_input, lon_input, lat_24, lon_24,
                                                      lat_48, lon_48, res["wind_speed"])
                cone_layer = pdk.Layer("ScatterplotLayer", data=cone, get_position=["lon", "lat"],
                                       get_radius="radius", get_fill_color=[255, 165, 0, 80])
                track_layer = pdk.Layer("ScatterplotLayer", data=track_df, get_position=["lon", "lat"],
                                        get_radius="size", get_fill_color="color",
                                        pickable=True, auto_highlight=True)
                view = pdk.ViewState(latitude=float(lat_input), longitude=float(lon_input), zoom=4)
                st.pydeck_chart(pdk.Deck(initial_view_state=view, layers=[cone_layer, track_layer],
                                         map_style="dark"))
                st.dataframe(track_df[["type", "lat", "lon"]], use_container_width=True)

        with tab4:
            st.subheader("📋 Automated Report")
            now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            report = f"""=== AI TYPHOON ANALYTICS CORE ===
> TIMESTAMP : {now}
> LOCATION  : {lat_input} N, {lon_input} E

[1] INTENSITY ESTIMATE (CNN)
    - Category     : {res['category']} - {res['lifecycle']}
    - Max Wind     : {res['wind_speed']} kt ({res['wind_kmh']} km/h)
    - Uncertainty  : ~{MODEL_INFO['test_rmse']} kt RMSE (test {MODEL_INFO['test_year']})
    - Input type   : {"approx. (PNG/JPG)" if res['approx_input'] else "IR brightness temp (.npy)"}

[2] MODEL
    - Architecture : {MODEL_INFO['arch']}
    - Training data: {MODEL_INFO['data']}
    - Val RMSE     : {MODEL_INFO['val_rmse']} kt

[3] TRACK MODULE
    - Method       : CLIPER-style extrapolation + beta drift (rule-based)
    - Horizon      : 48h

>> For research/demo purposes only. Not an official forecast.
"""
            st.code(report, language="yaml")
            st.download_button("💾 DOWNLOAD REPORT (.TXT)", data=report,
                               file_name="Storm_Report.txt", mime="text/plain")
    else:
        st.info("Upload an IR image and run analysis to see results.")
