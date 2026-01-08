import streamlit as st
from PIL import Image
import os
import sys
import pandas as pd
import numpy as np
import time

# --- 1. SETUP ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from source.models.intensity_model import IntensityRegressionModel
    from source.inference.predictor import IntensityPredictor
    from config import settings
except ImportError as e:
    st.error(f"❌ Lỗi Import: {e}")
    st.stop()

# --- 2. CẤU HÌNH GIAO DIỆN DARK MODE ---
st.set_page_config(page_title="Hệ thống Cảnh báo Bão AI", page_icon="🌪️", layout="wide")

# CSS Ép màu đen toàn tập
st.markdown("""
    <style>
    /* 1. Nền đen */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* 2. Chỉnh màu chữ thành trắng */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #ffffff !important;
    }
    
    /* 3. Chỉnh các ô nhập liệu (Input) */
    .stNumberInput input, .stTextInput input {
        color: white;
        background-color: #333333;
    }
    
    /* 4. Nút bấm đẹp */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border: 1px solid #ff4b4b;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        border-color: white;
    }
    
    /* 5. Khung Metric kết quả */
    div[data-testid="stMetricValue"] {
        color: #00ff00 !important; /* Số kết quả màu xanh lá cho nổi */
    }
    div[data-testid="stMetricLabel"] {
        color: #dddddd !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HÀM LOAD MODEL ---
@st.cache_resource
def load_ai_engine():
    model_path = os.path.join(BASE_DIR, "outputs", "best_model.pth")
    if not os.path.exists(model_path):
        return None, "⚠️ Chưa có file model! Hãy copy 'best_model.pth' vào folder 'outputs'."
    try:
        model = IntensityRegressionModel(backbone='resnet50', input_channels=3, pretrained=False)
        predictor = IntensityPredictor(model, checkpoint_path=model_path)
        return predictor, None
    except Exception as e:
        return None, str(e)

# --- 4. HÀM TÍNH TOÁN ĐƯỜNG ĐI (QUÁN TÍNH) ---
def predict_track(lat_old, lon_old, lat_curr, lon_curr):
    """
    Dự báo vị trí tương lai dựa trên vector di chuyển
    """
    delta_lat = lat_curr - lat_old
    delta_lon = lon_curr - lon_old
    
    future_points = []
    # Điểm hiện tại
    future_points.append({"lat": lat_curr, "lon": lon_curr, "type": "Hiện tại", "time": "+0h"})
    
    # Dự báo 24h tới
    for i in range(1, 5): 
        next_lat = lat_curr + (delta_lat * i)
        next_lon = lon_curr + (delta_lon * i)
        future_points.append({
            "lat": next_lat,
            "lon": next_lon,
            "type": "Dự báo",
            "time": f"+{i*6}h"
        })
        
    return pd.DataFrame(future_points)

# =========================================================
# GIAO DIỆN CHÍNH
# =========================================================

st.title("🌪️ AI CYCLONE TRACKER")
st.caption("Deep Learning Core: ResNet50 | Algorithm: Persistence Forecast")
st.markdown("---")

left_col, right_col = st.columns([1, 1.5])

# --- CỘT TRÁI: AI DỰ ĐOÁN ---
with left_col:
    st.subheader("📡 Phân tích ảnh vệ tinh")
    
    predictor, err = load_ai_engine()
    if err: st.error(err)
    
    uploaded_file = st.file_uploader("Upload ảnh vệ tinh (IR):", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        # Hiển thị ảnh (Đã sửa code cũ lỗi)
        st.image(uploaded_file, caption="Input Satellite Image", use_column_width=True)
        
        if st.button("🚀 PHÂN TÍCH (ANALYZE)"):
            if predictor:
                with st.spinner("AI Computing..."):
                    time.sleep(1) 
                    result = predictor.predict(Image.open(uploaded_file))
                
                # --- PHẦN TÍNH TOÁN ĐỔI ĐƠN VỊ ---
                knots = result['wind_speed']
                kmh = round(knots * 1.852, 1) # Công thức chuẩn: 1 kts = 1.852 km/h
                
                # Hiển thị kết quả
                st.success("Analysis Complete!")
                
                # --- CHIA THÀNH 3 CỘT (Knots | Km/h | Phân loại) ---
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.metric("Sức gió (Knots)", f"{knots} kts")
                
                with c2:
                    # Hiển thị Km/h màu xanh cho nổi bật
                    st.metric("Sức gió (Km/h)", f"{kmh} km/h", delta="VN Standard")
                    
                with c3:
                    st.metric("Phân loại", result['lifecycle'])
                
                # Box màu cảnh báo (Giữ nguyên)
                st.markdown(f"""
                <div style="background:{result.get('color','gray')}; padding:15px; border-radius:8px; color:white; text-align:center; font-weight:bold; font-size:18px; border: 2px solid white; margin-top: 10px;">
                    {result['lifecycle'].upper()}
                </div>
                """, unsafe_allow_html=True)
                
                st.write("Intensity Index:")
                st.progress(min(knots/150, 1.0))

# --- CỘT PHẢI: BẢN ĐỒ (ĐÃ FIX LỖI) ---
with right_col:
    st.subheader("🗺️ Dự báo đường đi (Track Map)")
    
    c_lat, c_lon = st.columns(2)
    with c_lat:
        st.write("📍 **Vị trí cũ (-6h):**")
        lat_prev = st.number_input("Vĩ độ (Lat)", value=15.5, key="lat1")
        lon_prev = st.number_input("Kinh độ (Lon)", value=111.0, key="lon1")
    
    with c_lon:
        st.write("📍 **Vị trí hiện tại (Now):**")
        lat_now = st.number_input("Vĩ độ (Lat)", value=16.2, key="lat2")
        lon_now = st.number_input("Kinh độ (Lon)", value=110.5, key="lon2")

    if st.button("📍 VẼ ĐƯỜNG ĐI (PROJECT TRACK)"):
        track_data = predict_track(lat_prev, lon_prev, lat_now, lon_now)
        
        st.write("### Simulation Result (24h)")
        
        # [FIX QUAN TRỌNG]: Đã xóa 'size=20000' để không bị lỗi trên máy bạn
        st.map(track_data, zoom=5)
        
        # Hiện bảng dữ liệu
        st.table(track_data[['time', 'lat', 'lon', 'type']])

st.markdown("---")
st.caption("System Status: ONLINE | GPU: Active")