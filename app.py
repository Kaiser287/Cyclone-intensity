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
# Bán kính vùng bất định (km) quanh mỗi điểm dự báo.
# TODO: thay bằng sai số trung bình LightGBM mà train_track.py in ra (tập test 2017+).
TRACK_CONE_KM = {24: 100, 48: 180, 72: 270}
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
        # track_lgbm.pkl được tự tìm trong cùng thư mục với intensity_best.pt
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
# --- 2. TRACK DISPLAY (dữ liệu dự báo lấy từ TrackPredictor qua predictor.py) ---
def norm_lon(lon):
    """TrackPredictor trả kinh độ 0..360; pydeck cần -180..180."""
    return float(lon - 360 if lon > 180 else lon)
def build_track_display(track, history):
    """track = res["track"], history = [(lat,lon) -48h, -24h, now] -> (df điểm, cone, path)."""
    (la48, lo48), (la24, lo24), (la0, lo0) = history
    rows = [
        {"lat": la48, "lon": norm_lon(lo48), "type": "History (-48h)", "size": 3000, "color": [100, 100, 100]},
        {"lat": la24, "lon": norm_lon(lo24), "type": "History (-24h)", "size": 4000, "color": [150, 150, 150]},
        {"lat": la0, "lon": norm_lon(lo0), "type": "CURRENT CENTER", "size": 12000, "color": [255, 0, 0]},
    ]
    cone = []
    for p in track["points"]:
        lat, lon, h = float(p["lat"]), norm_lon(p["lon"]), p["hour"]
        rows.append({"lat": lat, "lon": lon, "type": f"Forecast (+{h}h)",
                     "size": 6000 + h * 50, "color": [255, 165, 0]})
        cone.append({"lat": lat, "lon": lon, "radius": TRACK_CONE_KM.get(h, 150) * 1000})
    path = [{"path": [[r["lon"], r["lat"]] for r in rows]}]
    return pd.DataFrame(rows), cone, path
# ================= MAIN INTERFACE =================
st.title("🌪️ AI TYPHOON ANALYTICS CORE")
st.caption("CNN Intensity Estimation (TCIR) | 3D Cloud-Top View | LightGBM Track Forecast (IBTrACS)")
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
    track_method = ("LightGBM" if predictor is not None and predictor.track.is_ai
                    else "Persistence (track_lgbm.pkl not found)")
    st.caption(f"Model: {MODEL_INFO['arch']}\n\n"
               f"Test RMSE ({MODEL_INFO['test_year']}): {MODEL_INFO['test_rmse']} kt\n\n"
               f"Track: {track_method}")
history = [(lat_48, lon_48), (lat_24, lon_24), (lat_input, lon_input)]
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
                with st.spinner("Running CNN inference (4x TTA) + track forecast..."):
                    res = predictor.predict(model_input, track_history=history)
                    res["track_history"] = history
                    st.session_state["result"] = res
                    st.session_state["img_cache"] = display_img
        except Exception as e:
            st.error(f"Cannot read file: {e}")
with col_right:
    res = st.session_state["result"]
    if res:
        # Sửa toạ độ sau khi đã phân tích -> chỉ chạy lại track, không chạy lại CNN
        if res.get("track_history") != history:
            res["track"] = predictor.track.predict(history, res["wind_speed"])
            res["track_history"] = history
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
            track = res["track"]
            if track["method"] == "LightGBM":
                st.write("LightGBM forecast trained on IBTrACS WPAC (1980–2014), "
                         "predicting position offsets at +24/48/72h.")
            else:
                st.warning("track_lgbm.pkl not found – showing persistence baseline "
                           "(constant 24h motion). Run train_track.py to enable the AI model.")
            st.info(f"Inputs: 48h position history + CNN intensity **{res['wind_speed']} kt**")
            track_df, cone, path = build_track_display(track, history)
            cone_layer = pdk.Layer("ScatterplotLayer", data=cone, get_position=["lon", "lat"],
                                   get_radius="radius", get_fill_color=[255, 165, 0, 60])
            path_layer = pdk.Layer("PathLayer", data=path, get_path="path",
                                   get_color=[255, 255, 255, 160], width_min_pixels=2)
            track_layer = pdk.Layer("ScatterplotLayer", data=track_df, get_position=["lon", "lat"],
                                    get_radius="size", get_fill_color="color",
                                    pickable=True, auto_highlight=True)
            view = pdk.ViewState(latitude=float(lat_input), longitude=float(lon_input), zoom=4)
            st.pydeck_chart(pdk.Deck(initial_view_state=view,
                                     layers=[cone_layer, path_layer, track_layer],
                                     map_style="dark", tooltip={"text": "{type}"}))
            st.dataframe(track_df[["type", "lat", "lon"]].round(2), use_container_width=True)
            st.caption("Orange circles: approximate uncertainty radius per lead time.")
        with tab4:
            st.subheader("📋 Automated Report")
            now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            fc_lines = "\n".join(
                f"    - +{p['hour']}h        : {p['lat']:.2f} N, {norm_lon(p['lon']):.2f} E"
                for p in res["track"]["points"])
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
[3] TRACK FORECAST
    - Method       : {res['track']['method']}{" (IBTrACS WPAC)" if res['track']['method'] == "LightGBM" else " baseline"}
    - Horizon      : 72h
{fc_lines}
>> For research/demo purposes only. Not an official forecast.
"""
            st.code(report, language="yaml")
            st.download_button("💾 DOWNLOAD REPORT (.TXT)", data=report,
                               file_name="Storm_Report.txt", mime="text/plain")
    else:
        st.info("Upload an IR image and run analysis to see results.")
