import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr

st.set_page_config(
    page_title="Lab Report OCR | AI-Powered Extraction",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main container gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        color: #667eea;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #666;
        font-size: 0.95rem;
    }
    
    /* Upload section */
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        margin: 2rem 0;
    }
    
    /* Results card */
    .results-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        margin: 2rem 0;
        animation: fadeIn 0.5s ease-out;
    }
    
    .section-title {
        color: #667eea;
        font-size: 1.8rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1.5rem;
        position: relative;
        padding-bottom: 0.5rem;
    }
    
    .section-title:after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Download button special styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(17, 153, 142, 0.5);
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Alert/Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed #667eea;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Progress indicator */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

@st.cache_resource
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'best.onnx' not found.")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    return model

def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1 / 255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

def process_predictions(preds, input_img, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences, class_ids = [], [], []
    detections = preds[0]
    h, w = input_img.shape[:2]
    x_factor = w / 640
    y_factor = h / 640
    for det in detections:
        conf = det[4]
        if conf > conf_thresh:
            scores = det[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > score_thresh:
                cx, cy, bw, bh = det[:4]
                x = int((cx - bw / 2) * x_factor)
                y = int((cy - bh / 2) * y_factor)
                boxes.append([x, y, int(bw * x_factor), int(bh * y_factor)])
                confidences.append(float(conf))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    return indices.flatten() if len(indices) > 0 else [], boxes, class_ids

def extract_table_text(image, boxes, indices, class_ids):
    reader = easyocr.Reader(["en"], gpu=False)
    results = {key: [] for key in class_map.values()}

    for i in indices:
        if i >= len(boxes) or i >= len(class_ids):
            continue
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Field")
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(binary)

        try:
            lines = reader.readtext(roi, detail=0)
        except Exception:
            lines = []

        for line in lines:
            clean = line.strip()
            if not clean:
                continue
            results[label].append(clean)

    max_len = max(len(v) for v in results.values()) if results else 0
    for k in results:
        results[k] += [""] * (max_len - len(results[k]))

    df = pd.DataFrame(results)
    return df

def draw_boxes(image, boxes, indices, class_ids):
    colors = {
        0: (102, 126, 234),  # Purple-blue
        1: (56, 239, 125),   # Green
        2: (255, 107, 107),  # Red
        3: (252, 196, 25)    # Yellow
    }
    
    for i in indices:
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Field")
        color = colors.get(class_ids[i], (0, 255, 0))
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
        cv2.putText(
            image,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    return image

st.markdown("""
<div class="main-header">
    <h1>🔬 Lab Report OCR Extractor</h1>
    <p>AI-Powered Medical Report Analysis & Data Extraction</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">AI Detection</div>
        <div class="feature-desc">YOLOv5 powered field detection</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📝</div>
        <div class="feature-title">Smart OCR</div>
        <div class="feature-desc">EasyOCR text extraction</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Structured Data</div>
        <div class="feature-desc">Organized table format</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">Fast Processing</div>
        <div class="feature-desc">Instant results</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; margin: 2rem 0;'>
    <div style='background: white; display: inline-block; padding: 1rem 2rem; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <span style='font-size: 1.1rem;'>📥 Need sample reports? </span>
        <a href='https://drive.google.com/drive/folders/1NnOcNYggvuU2T9YzCiYO4biKVdwHmz_9?usp=drive_link' 
           target='_blank' 
           style='color: #667eea; font-weight: 600; text-decoration: none;'>
           Download from Drive →
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #667eea; margin-bottom: 1rem;'>📤 Upload Lab Reports</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; margin-bottom: 1.5rem;'>Supported formats: JPG, JPEG, PNG</p>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Choose files",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=st.session_state.get("uploader_key", "file_uploader"),
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_files:
    model = load_yolo_model()
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(uploaded_files)}</div>
            <div class="stat-label">Files Uploaded</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">4</div>
            <div class="stat-label">Field Types</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">✓</div>
            <div class="stat-label">AI Ready</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    for idx, file in enumerate(uploaded_files, 1):
        st.markdown(f"""
        <div class='results-card'>
            <h3 style='color: #667eea; text-align: center; margin-bottom: 1.5rem;'>
                📄 Report {idx}: {file.name}
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Loading image...")
        progress_bar.progress(20)
        image = np.array(Image.open(file).convert("RGB"))
        
        status_text.text("🔍 Running AI detection...")
        progress_bar.progress(40)
        preds, input_img = predict_yolo(model, image)
        
        status_text.text("📦 Processing detections...")
        progress_bar.progress(60)
        indices, boxes, class_ids = process_predictions(preds, input_img)
        
        if len(indices) == 0:
            progress_bar.empty()
            status_text.empty()
            st.warning("⚠️ No fields detected in this image. Please try another image.")
            continue
        
        status_text.text("📝 Extracting text with OCR...")
        progress_bar.progress(80)
        df = extract_table_text(image, boxes, indices, class_ids)
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        st.markdown("""
        <div class='results-card'>
            <div class='section-title'>📊 Extracted Data Table</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            df,
            use_container_width=True,
            height=300
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='results-card'>
                <div class='section-title'>🖼️ Detected Fields Visualization</div>
            </div>
            """, unsafe_allow_html=True)
            annotated_img = draw_boxes(image.copy(), boxes, indices, class_ids)
            st.image(annotated_img, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class='results-card'>
                <div class='section-title'>📈 Detection Statistics</div>
            </div>
            """, unsafe_allow_html=True)
            
            stats_df = pd.DataFrame({
                'Field Type': [class_map[i] for i in range(4)],
                'Count': [sum(1 for cid in class_ids if cid == i) for i in range(4)]
            })
            
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            st.markdown("""
            <div style='margin-top: 1.5rem; padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px;'>
                <strong style='color: #667eea;'>Legend:</strong><br>
                <span style='color: #667eea;'>■</span> Test Name &nbsp;
                <span style='color: #38ef7d;'>■</span> Value<br>
                <span style='color: #ff6b6b;'>■</span> Units &nbsp;
                <span style='color: #fcc419;'>■</span> Reference Range
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.download_button(
                "⬇️ Download CSV",
                df.to_csv(index=False),
                file_name=f"{file.name}_extracted.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                "📊 Download Excel",
                df.to_csv(index=False),
                file_name=f"{file.name}_extracted.xlsx",
                mime="application/vnd.ms-excel",
                use_container_width=True
            )
        
        with col3:
            if st.button("🧹 Clear All", key=f"clear_{idx}", use_container_width=True):
                st.session_state["uploaded_files"] = []
                st.session_state["extracted_dfs"] = []
                st.session_state["uploader_key"] = "file_uploader_" + str(np.random.randint(1_000_000))
                st.rerun()
        
        st.markdown("<hr style='margin: 3rem 0; border: none; border-top: 2px solid rgba(102, 126, 234, 0.2);'>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-top: 4rem; padding: 2rem; background: white; border-radius: 20px; box-shadow: 0 5px 20px rgba(0,0,0,0.1);'>
    <p style='color: #667eea; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;'>
        🔬 Lab Report OCR Extractor
    </p>
    <p style='color: #666; font-size: 0.9rem;'>
        Powered by YOLOv5 & EasyOCR | Made with using Streamlit
    </p>
</div>
""", unsafe_allow_html=True)