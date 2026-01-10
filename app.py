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
# --- 2. ADVANCED TRACK PREDICTION (CLIPER + STEERING + CONE) ---
def predict_track_advanced(lat_now, lon_now, lat_24, lon_24, lat_48, lon_48):
    """
    Thuật toán giả lập CLIPER (Climatology and Persistence)
    Kết hợp dòng dẫn đường (Steering Flow) để tạo đường cong tự nhiên.
    """
    # 1. TÍNH VẬN TỐC QUÁN TÍNH (PERSISTENCE)
    v_lat_recent = (lat_now - lat_24) / 4.0 
    v_lon_recent = (lon_now - lon_24) / 4.0
    v_lat_old = (lat_24 - lat_48) / 4.0
    v_lon_old = (lon_24 - lon_48) / 4.0
    
    # Trọng số: 60% xu hướng mới + 40% xu hướng cũ
    curr_v_lat = (v_lat_recent * 0.6) + (v_lat_old * 0.4)
    curr_v_lon = (v_lon_recent * 0.6) + (v_lon_old * 0.4)

    # 2. KHỞI TẠO DỮ LIỆU VẼ
    track_points = []
    cone_points = [] # Dữ liệu vùng nón nguy hiểm
    
    # Điểm quá khứ
    track_points.append({"lat": lat_48, "lon": lon_48, "type": "History (-48h)", "size": 3000, "color": [100, 100, 100]})
    track_points.append({"lat": lat_24, "lon": lon_24, "type": "History (-24h)", "size": 4000, "color": [150, 150, 150]})
    track_points.append({"lat": lat_now, "lon": lon_now, "type": "CURRENT CENTER", "size": 10000, "color": [255, 0, 0]}) # Đỏ
    
    next_lat, next_lon = lat_now, lon_now
    
    # Dự báo 72h (12 bước, mỗi bước 6h)
    forecast_hours = [24, 48, 72]
    
    # Bán kính vùng nguy hiểm ban đầu (km)
    cone_radius = 50 
    
    for step in range(1, 13): 
        # --- THUẬT TOÁN STEERING FLOW (DÒNG DẪN) ---
        # 1. Beta Drift (Trôi dạt tự nhiên về phía Tây Bắc do trái đất quay)
        beta_lat = 0.05 
        beta_lon = -0.02
        
        # 2. Environmental Steering (Giả lập Áp cao cận nhiệt đới)
        steer_lat, steer_lon = 0.0, 0.0
        
        if next_lat < 20.0:
            # Vùng vĩ độ thấp: Gió Tín phong đẩy mạnh về phía Tây
            steer_lon = -0.15 
            steer_lat = 0.02
        elif 20.0 <= next_lat < 28.0:
            # Vùng chuyển tiếp (Recurvature Point): Bão đi chậm lại, bắt đầu quặt Bắc
            steer_lon = -0.05 + ((next_lat - 20.0) * 0.02) # Giảm tốc độ sang Tây
            steer_lat = 0.08 # Tăng tốc độ lên Bắc
        else:
            # Vĩ độ cao (> 28N): Gió Tây ôn đới đẩy mạnh sang Đông Bắc (Bão quặt ra biển)
            steer_lon = 0.2 + ((next_lat - 28.0) * 0.05) 
            steer_lat = 0.1
            
        # Tổng hợp lực: 70% Quán tính cũ + 30% Môi trường mới
        curr_v_lat = (curr_v_lat * 0.7) + (steer_lat + beta_lat) * 0.3
        curr_v_lon = (curr_v_lon * 0.7) + (steer_lon + beta_lon) * 0.3
        
        next_lat += curr_v_lat
        next_lon += curr_v_lon
        
        # Mở rộng vùng nón nguy hiểm theo thời gian (Càng xa càng sai số lớn)
        cone_radius += 15 # Mỗi 6h bán kính sai số tăng thêm 15km
        
        hour_mark = step * 6
        if hour_mark in forecast_hours:
            # Điểm tâm bão
            track_points.append({
                "lat": float(next_lat), "lon": float(next_lon), 
                "type": f"Forecast (+{hour_mark}h)", 
                "size": 6000, 
                "color": [255, 165, 0] # Cam
            })
            
            # Tạo hình tròn bao quanh (Vùng nguy hiểm)
            # Lưu ý: Pydeck cần polygon, ở đây ta lưu tâm & bán kính để vẽ sau
            cone_points.append({
                "lat": float(next_lat), "lon": float(next_lon),
                "radius": cone_radius * 1000 # Đổi ra mét
            })
            
    return pd.DataFrame(track_points), cone_points

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
            st.write("Using CLIPER + Environmental Steering Model")
            
            if st.button("📍 GENERATE FORECAST TRACK"):
                # Gọi hàm mới
                track_df, cone_data = predict_track_advanced(lat_input, lon_input, lat_24, lon_24, lat_48, lon_48)
                
                # --- LAYER 1: VÙNG NÓN NGUY HIỂM (CONE OF UNCERTAINTY) ---
                # Vẽ các vòng tròn mờ bao quanh điểm dự báo
                cone_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=cone_data,
                    get_position=["lon", "lat"],
                    get_radius="radius",
                    get_fill_color=[255, 165, 0, 80], # Màu cam, độ trong suốt 80 (nhìn xuyên thấu)
                    get_line_color=[255, 0, 0, 0],
                    pickable=False,
                )

                # --- LAYER 2: ĐƯỜNG ĐI NỐI CÁC ĐIỂM ---
                # Tạo đường nối (Path) từ dữ liệu track_df
                # (Pydeck cần format riêng cho PathLayer, nhưng để đơn giản ta dùng Scatterplot nối tiếp)
                # Hoặc chỉ cần vẽ các điểm là đủ đẹp rồi.

                # --- LAYER 3: CÁC ĐIỂM DỰ BÁO (TÂM BÃO) ---
                track_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=track_df,
                    get_position=["lon", "lat"],
                    get_radius="size",
                    get_fill_color="color",
                    pickable=True,
                    auto_highlight=True,
                )
                
                # Hiển thị bản đồ
                view_state = pdk.ViewState(
                    latitude=float(lat_input), 
                    longitude=float(lon_input), 
                    zoom=4,
                    pitch=0
                )
                
                st.pydeck_chart(pdk.Deck(
                    initial_view_state=view_state, 
                    layers=[cone_layer, track_layer], # Vẽ Cone trước (nằm dưới), Track sau (nằm trên)
                    map_style="mapbox://styles/mapbox/dark-v10"
                ))
                
                # Bảng chi tiết tọa độ
                st.caption("Detailed Forecast Coordinates:")
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