import streamlit as st
from PIL import Image
import os
import sys
import pandas as pd
import numpy as np
import time
import pydeck as pdk # Vẽ 3D

# --- SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from source.models.intensity_model import IntensityRegressionModel
    from source.inference.predictor import IntensityPredictor
except ImportError as e:
    st.error(f"❌ Lỗi Import: {e}")
    st.stop()

# --- CẤU HÌNH DARK MODE ---
st.set_page_config(page_title="AI Typhoon Center", page_icon="🌪️", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1, h2, h3 { color: #ffffff !important; }
    .stButton>button { background-color: #d93025; color: white; font-weight: bold; border: none; height: 50px; width: 100%; }
    div[data-testid="stMetricValue"] { color: #4caf50 !important; font-size: 26px !important; }
    .report-box { background-color: #1e1e1e; padding: 20px; border-radius: 5px; border-left: 5px solid #d93025; font-family: monospace; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_ai_engine():
    model_path = os.path.join(BASE_DIR, "outputs", "best_model.pth")
    if not os.path.exists(model_path): return None, "Thiếu file best_model.pth"
    try:
        model = IntensityRegressionModel(backbone='resnet50', input_channels=3, pretrained=False)
        predictor = IntensityPredictor(model, checkpoint_path=model_path)
        return predictor, None
    except Exception as e: return None, str(e)

# --- 1. THUẬT TOÁN VẼ 3D (Đã Fix lỗi Numpy cho Python 3.7) ---
def generate_3d_storm_data(image, center_lat, center_lon):
    img_small = image.resize((60, 60)).convert('L') # Resize để vẽ mượt
    pixels = np.array(img_small)
    data = []
    lat_step, lon_step = 0.05, 0.05
    rows, cols = pixels.shape
    
    # [FIX QUAN TRỌNG]: Ép kiểu center_lat/lon về float chuẩn
    center_lat = float(center_lat)
    center_lon = float(center_lon)

    for r in range(rows):
        for c in range(cols):
            # [FIX QUAN TRỌNG]: Ép kiểu numpy.uint8 -> int chuẩn của Python
            brightness = int(pixels[r, c]) 
            
            if brightness > 40: # Chỉ vẽ mây rõ
                # Ép kiểu float chuẩn
                height = float((brightness / 255.0) * 70000) 
                
                # Tính toán màu sắc (ép kiểu int)
                color = [int(brightness), int(brightness), int(255-brightness), 200]
                
                # Tính tọa độ (ép kiểu float)
                lat = float(center_lat - (r - rows/2) * lat_step)
                lon = float(center_lon + (c - cols/2) * lon_step)
                
                data.append({"lat": lat, "lon": lon, "height": height, "color": color})
    return pd.DataFrame(data)

# --- 2. THUẬT TOÁN ĐƯỜNG ĐI (CLIPER + Coriolis) ---
def predict_track_advanced(lat_old, lon_old, lat_curr, lon_curr):
    # Ép kiểu float đầu vào để tránh lỗi
    lat_curr, lon_curr = float(lat_curr), float(lon_curr)
    lat_old, lon_old = float(lat_old), float(lon_old)

    v_lat = lat_curr - lat_old
    v_lon = lon_curr - lon_old
    
    points = []
    points.append({"lat": lat_old, "lon": lon_old, "type": "Quá khứ (-6h)", "size": 5000, "color": [100, 100, 100]})
    points.append({"lat": lat_curr, "lon": lon_curr, "type": "TÂM BÃO (Hiện tại)", "size": 10000, "color": [255, 0, 0]})
    
    next_lat, next_lon = lat_curr, lon_curr
    curr_v_lat, curr_v_lon = v_lat, v_lon
    
    # Dự báo 12 bước (72h)
    for i in range(1, 13): 
        beta_lat = 0.05 
        beta_lon = -0.02
        
        steer_lat, steer_lon = 0.0, 0.0
        if next_lat > 20.0:
            steer_lon = 0.08 * (next_lat - 20.0)
            steer_lat = 0.02
        
        curr_v_lat = (curr_v_lat * 0.9) + beta_lat + steer_lat
        curr_v_lon = (curr_v_lon * 0.9) + beta_lon + steer_lon
        
        next_lat += curr_v_lat
        next_lon += curr_v_lon
        
        points.append({
            "lat": float(next_lat), "lon": float(next_lon), 
            "type": f"Dự báo (+{i*6}h)", "size": 5000, "color": [255, 165, 0]
        })
    return pd.DataFrame(points)

# ================= GIAO DIỆN CHÍNH =================
st.title("🇻🇳 TRUNG TÂM CẢNH BÁO BÃO AI (VN-S)")
st.caption("Deep Learning Core | 3D Visualization | CLIPER Track Forecast")
st.markdown("---")

# Session State
if 'result' not in st.session_state: st.session_state['result'] = None
if 'img_cache' not in st.session_state: st.session_state['img_cache'] = None

col_left, col_right = st.columns([1, 1.5])

# === CỘT TRÁI: INPUT & AI ===
with col_left:
    st.subheader("1. Dữ liệu Vệ tinh")
    predictor, err = load_ai_engine()
    if err: st.error(err)
    
    uploaded_file = st.file_uploader("Upload ảnh vệ tinh (IR):", type=["jpg", "png"])
    
    c1, c2 = st.columns(2)
    with c1: lat_input = st.number_input("Vĩ độ Tâm bão", 16.0)
    with c2: lon_input = st.number_input("Kinh độ Tâm bão", 110.0)
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Ảnh vệ tinh gốc", use_column_width=True)
        if st.button("🚀 KÍCH HOẠT PHÂN TÍCH"):
            with st.spinner("AI đang tính toán (Gió, Mưa, Cấu trúc)..."):
                time.sleep(1)
                st.session_state['result'] = predictor.predict(img)
                st.session_state['img_cache'] = img

# === CỘT PHẢI: KẾT QUẢ ĐA CHIỀU ===
with col_right:
    if st.session_state['result']:
        res = st.session_state['result']
        kmh = round(res['wind_speed'] * 1.852, 1)
        
        # TAB GIAO DIỆN
        tab1, tab2, tab3, tab4 = st.tabs(["📊 THÔNG SỐ", "🧊 CẤU TRÚC 3D", "🗺️ ĐƯỜNG ĐI", "📜 CÔNG ĐIỆN"])
        
        with tab1: # Thông số
            r1c1, r1c2 = st.columns(2)
            r1c1.metric("Sức gió (Km/h)", f"{kmh}", delta="Chuẩn VN")
            r1c2.metric("Sức gió (Knots)", f"{res['wind_speed']}")
            
            r2c1, r2c2 = st.columns(2)
            r2c1.metric("Lượng mưa (1h)", f"{res['rainfall']} mm")
            r2c2.markdown(f"<div style='color:{res['color']}; font-weight:bold; font-size:20px'>{res['lifecycle']}</div>", unsafe_allow_html=True)
            st.progress(min(res['wind_speed']/180, 1.0))

        with tab2: # 3D Map
            st.info("💡 Xoay chuột để thấy tường mắt bão dựng đứng!")
            # Gọi hàm đã fix lỗi
            df_3d = generate_3d_storm_data(st.session_state['img_cache'], lat_input, lon_input)
            
            layer = pdk.Layer(
                "ColumnLayer", data=df_3d, get_position=["lon", "lat"],
                get_elevation="height", elevation_scale=1, radius=4000,
                get_fill_color="color", pickable=True, auto_highlight=True,
            )
            view_state = pdk.ViewState(latitude=float(lat_input), longitude=float(lon_input), zoom=6, pitch=60)
            st.pydeck_chart(pdk.Deck(initial_view_state=view_state, layers=[layer]))

        with tab3: # Đường đi
            st.write("Dự báo quỹ đạo (Mô hình CLIPER + Beta Drift)")
            c_lat, c_lon = st.columns(2)
            with c_lat: lat_old = st.number_input("Vĩ độ cũ (-6h)", float(lat_input) - 0.5)
            with c_lon: lon_old = st.number_input("Kinh độ cũ (-6h)", float(lon_input) + 0.5)
            
            if st.button("📍 VẼ ĐƯỜNG ĐI"):
                track_df = predict_track_advanced(lat_old, lon_old, lat_input, lon_input)
                # Dùng tham số cơ bản cho st.map để tránh lỗi phiên bản cũ
                st.map(track_df, zoom=4) 
                # Hiển thị chú thích màu riêng
                st.caption("🔴 Đỏ: Hiện tại | 🟠 Cam: Dự báo | ⚫ Xám: Quá khứ")

        with tab4: # Báo cáo
            report = f"""
            === CÔNG ĐIỆN KHẨN (TỰ ĐỘNG) ===
            THỜI GIAN: {time.strftime("%H:%M %d/%m/%Y")}
            VỊ TRÍ TÂM: {lat_input}N - {lon_input}E
            --------------------------------
            1. HIỆN TRẠNG:
               - Cường độ: {res['lifecycle'].upper()}
               - Gió mạnh nhất: {kmh} km/h (Cấp {int(kmh/15)+6})
               - Mưa dự báo: {res['rainfall']} mm/h
            2. DỰ BÁO:
               - Bão di chuyển phức tạp theo hướng Tây Bắc.
               - Nguy cơ ngập lụt vùng tâm bão đi qua: CAO.
            --------------------------------
            SYSTEM: AI STORM SENTINEL V1.0
            """
            st.markdown(f"<div class='report-box'><pre>{report}</pre></div>", unsafe_allow_html=True)
            
    else:
        st.info("👈 Vui lòng Upload ảnh và chạy hệ thống.")