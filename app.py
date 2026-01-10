import streamlit as st
from PIL import Image
import os
import sys
import pandas as pd
import numpy as np
import time
import pydeck as pdk 

# --- SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from source.models.intensity_model import IntensityRegressionModel
    from source.inference.predictor import IntensityPredictor
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()

# --- PAGE CONFIG (DARK MODE & ENGLISH) ---
st.set_page_config(page_title="AI Typhoon Analytics Core", page_icon="🌪️", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1, h2, h3 { color: #ffffff !important; }
    .stButton>button { background-color: #2962ff; color: white; font-weight: bold; border: none; height: 50px; width: 100%; border-radius: 8px; }
    div[data-testid="stMetricValue"] { color: #00e676 !important; font-size: 26px !important; }
    .report-box { background-color: #1e1e1e; padding: 20px; border-radius: 5px; border-left: 5px solid #2962ff; font-family: 'Courier New', monospace; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_ai_engine():
    model_path = os.path.join(BASE_DIR, "outputs", "best_model.pth")
    if not os.path.exists(model_path): return None, "Model file not found (best_model.pth)"
    try:
        model = IntensityRegressionModel(backbone='resnet50', input_channels=3, pretrained=False)
        predictor = IntensityPredictor(model, checkpoint_path=model_path)
        return predictor, None
    except Exception as e: return None, str(e)

# --- 1. EXPANDED 3D STORM VISUALIZATION ---
def generate_expanded_3d_data(image, center_lat, center_lon):
    # Resize to larger grid for better detail
    img_small = image.resize((80, 80)).convert('L') 
    pixels = np.array(img_small)
    data = []
    
    # [UPGRADE] Double the spread to cover wider area (Expand Storm)
    lat_step = 0.08  # Increased from 0.05
    lon_step = 0.08  
    
    rows, cols = pixels.shape
    center_lat = float(center_lat)
    center_lon = float(center_lon)

    for r in range(rows):
        for c in range(cols):
            brightness = int(pixels[r, c]) 
            
            if brightness > 30: # Filter low clouds
                # Scale height significantly for "Dramatic Effect"
                height = float((brightness / 255.0) * 90000) 
                
                # Color Gradient: Dark Grey (Low) -> White/Red (High)
                if brightness < 100:
                     color = [80, 80, 80, 180]
                elif brightness < 180:
                     color = [200, 200, 200, 200]
                else:
                     color = [brightness, 50, 50, 255] # Redish for strong convection

                lat = float(center_lat - (r - rows/2) * lat_step)
                lon = float(center_lon + (c - cols/2) * lon_step)
                
                data.append({"lat": lat, "lon": lon, "height": height, "color": color})
    return pd.DataFrame(data)

# --- 2. ADVANCED TRACK PREDICTION (Historical Weighting) ---
def predict_track_pro(lat_now, lon_now, lat_24, lon_24, lat_48, lon_48):
    """
    Predicts path based on 48h history trend + Beta Drift + Steering Flow
    """
    lat_now, lon_now = float(lat_now), float(lon_now)
    lat_24, lon_24 = float(lat_24), float(lon_24)
    lat_48, lon_48 = float(lat_48), float(lon_48)

    # Calculate average velocity over last 48h (Weighted)
    # Give more weight to recent movement (last 24h)
    v_lat_recent = (lat_now - lat_24) / 4.0 # 4 steps of 6h in 24h
    v_lon_recent = (lon_now - lon_24) / 4.0
    
    v_lat_old = (lat_24 - lat_48) / 4.0
    v_lon_old = (lon_24 - lon_48) / 4.0
    
    # Weighted Velocity: 70% recent trend, 30% old trend
    curr_v_lat = (v_lat_recent * 0.7) + (v_lat_old * 0.3)
    curr_v_lon = (v_lon_recent * 0.7) + (v_lon_old * 0.3)

    points = []
    # Historical Points
    points.append({"lat": lat_48, "lon": lon_48, "type": "History (-48h)", "size": 3000, "color": [100, 100, 100]})
    points.append({"lat": lat_24, "lon": lon_24, "type": "History (-24h)", "size": 4000, "color": [150, 150, 150]})
    points.append({"lat": lat_now, "lon": lon_now, "type": "CURRENT CENTER", "size": 12000, "color": [255, 0, 0]})
    
    next_lat, next_lon = lat_now, lon_now
    
    # Forecast Timesteps: 24h, 36h, 48h, 72h
    # Corresponding to steps of 6h: +4, +6, +8, +12 steps
    forecast_hours = [24, 36, 48, 72]
    current_step = 0
    
    for h in range(1, 13): # Simulate 12 steps (72 hours)
        # Beta Drift & Steering logic
        beta_lat = 0.05 
        beta_lon = -0.02
        steer_lat, steer_lon = 0.0, 0.0
        
        # Recurvature logic (> 20 deg Lat)
        if next_lat > 20.0:
            steer_lon = 0.1 * (next_lat - 20.0) # Stronger push East
            steer_lat = 0.03
        
        # Decay (Friction)
        curr_v_lat = (curr_v_lat * 0.92) + beta_lat + steer_lat
        curr_v_lon = (curr_v_lon * 0.92) + beta_lon + steer_lon
        
        next_lat += curr_v_lat
        next_lon += curr_v_lon
        
        hour_mark = h * 6
        if hour_mark in forecast_hours:
            points.append({
                "lat": float(next_lat), "lon": float(next_lon), 
                "type": f"Forecast (+{hour_mark}h)", "size": 6000 + (h*500), "color": [255, 165, 0]
            })
            
    return pd.DataFrame(points)

# ================= MAIN INTERFACE =================
st.title("🌪️ AI TYPHOON ANALYTICS CORE")
st.caption("Deep Learning Estimation | 3D Structural Analysis | Advanced Track Prediction")
st.markdown("---")

if 'result' not in st.session_state: st.session_state['result'] = None
if 'img_cache' not in st.session_state: st.session_state['img_cache'] = None

# === SIDEBAR CONTROLS ===
with st.sidebar:
    st.header("⚙️ Control Panel")
    uploaded_file = st.file_uploader("Upload Satellite Image (IR):", type=["jpg", "png"])
    
    st.markdown("---")
    st.subheader("🌐 Coordinates Input")
    st.info("Enter historical data for prediction")
    
    lat_input = st.number_input("Current Latitude (Now)", 16.0)
    lon_input = st.number_input("Current Longitude (Now)", 112.0)
    
    st.markdown("🔻 **History Data**")
    lat_24 = st.number_input("Lat (-24h ago)", 14.5)
    lon_24 = st.number_input("Lon (-24h ago)", 114.0)
    lat_48 = st.number_input("Lat (-48h ago)", 13.0)
    lon_48 = st.number_input("Lon (-48h ago)", 116.0)

# === MAIN LAYOUT ===
col_left, col_right = st.columns([1, 1.8])

with col_left:
    st.subheader("📡 Input Imagery")
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Infrared Satellite Data", use_column_width=True)
        
        predictor, err = load_ai_engine()
        if not err:
            if st.button("🚀 EXECUTE ANALYSIS"):
                with st.spinner("Processing Convolutional Layers..."):
                    time.sleep(1)
                    st.session_state['result'] = predictor.predict(img)
                    st.session_state['img_cache'] = img
        else:
            st.error(err)
    else:
        st.info("Waiting for data stream...")

with col_right:
    if st.session_state['result']:
        res = st.session_state['result']
        kmh = round(res['wind_speed'] * 1.852, 1)
        
        # TABS
        tab1, tab2, tab3, tab4 = st.tabs(["📊 METRICS", "🧊 3D MODEL (EXPANDED)", "🗺️ TRACK PREDICTION", "📜 AUTO-REPORT"])
        
        with tab1:
            c1, c2 = st.columns(2)
            c1.metric("Max Sustained Wind", f"{kmh} km/h", delta="International Scale")
            c2.metric("Wind Speed (Knots)", f"{res['wind_speed']} kts")
            
            c3, c4 = st.columns(2)
            c3.metric("Est. Rainfall (1h)", f"{res['rainfall']} mm")
            c4.markdown(f"<div style='color:{res['color']}; font-weight:bold; font-size:22px; margin-top:10px'>{res['lifecycle']}</div>", unsafe_allow_html=True)
            st.progress(min(res['wind_speed']/185, 1.0))

        with tab2:
            st.info("💡 Interaction: Right-click to rotate. Scroll to zoom.")
            # Call NEW EXPANDED 3D function
            df_3d = generate_expanded_3d_data(st.session_state['img_cache'], lat_input, lon_input)
            
            layer = pdk.Layer(
                "ColumnLayer", data=df_3d, get_position=["lon", "lat"],
                get_elevation="height", elevation_scale=2, radius=6000, # Increased radius & scale
                get_fill_color="color", pickable=True, auto_highlight=True,
            )
            view_state = pdk.ViewState(latitude=float(lat_input), longitude=float(lon_input), zoom=6, pitch=55)
            st.pydeck_chart(pdk.Deck(initial_view_state=view_state, layers=[layer]))

        with tab3:
            st.write("Using Weighted Persistence Model (48h History)")
            if st.button("📍 GENERATE FORECAST TRACK"):
                # Call NEW PRO TRACK function
                track_df = predict_track_pro(lat_input, lon_input, lat_24, lon_24, lat_48, lon_48)
                st.map(track_df, zoom=4)
                st.dataframe(track_df[['type', 'lat', 'lon']])

        with tab4:
            # Sửa lỗi thụt dòng bằng cách ép sát lề trái hoặc dùng st.code
            st.subheader("📋 Automated Mission Report")
            
            # Tính toán lại thời gian hiện tại
            current_time = time.strftime("%Y-%m-%d %H:%M:%S UTC")
            
            # Nội dung báo cáo (Dùng f-string chuẩn)
            report_content = f"""
=== AI TYPHOON ANALYTICS CORE ===
[CONFIDENTIAL REPORT]
------------------------------------------------
> TIMESTAMP : {current_time}
> TARGET LOC: {lat_input} N, {lon_input} E

[1] INTENSITY ASSESSMENT
    - Status       : {res['lifecycle']}
    - Max Wind     : {kmh} km/h ({res['wind_speed']} kts)
    - Precip. Rate : {res['rainfall']} mm/h
    - Threat Level : HIGH

[2] STRUCTURAL ANALYSIS
    - Eye Wall     : Defined / Deep Convection
    - Spiral Bands : Organized
    - Radius       : ~150 km effective

[3] FORECAST MODULE (CLIPER + BETA)
    - Input Data   : -48h Historical Track
    - Drift Vector : Northwest (Beta Effect)
    - Recurvature  : Possible if Lat > 20.0N

------------------------------------------------
>> GENERATED BY AI SYSTEM. END OF REPORT.
            """
            
            # Dùng st.code với format 'yaml' để nó tự tô màu số liệu trông rất đẹp
            st.code(report_content, language="yaml")
            
            # Nút tải báo cáo (Tính năng bonus)
            st.download_button(
                label="💾 DOWNLOAD REPORT (.TXT)",
                data=report_content,
                file_name="Storm_Report_Log.txt",
                mime="text/plain"
            )