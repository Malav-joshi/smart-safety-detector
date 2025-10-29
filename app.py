import streamlit as st
from ultralytics import YOLO
import tempfile
import os

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Smart Safety Detector",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================== CUSTOM STYLES ======================
st.markdown("""
    <style>
        .main {
            background-color: #f8fafc;
            font-family: "Segoe UI", sans-serif;
        }
        h1, h2, h3 {
            color: #2b2d42;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #1e40af;
        }
        .stAlert {
            border-radius: 10px;
        }
        .footer {
            text-align: center;
            margin-top: 2rem;
            font-size: 0.9rem;
            color: #6b7280;
        }
    </style>
""", unsafe_allow_html=True)

# ====================== MODEL LOADING ======================
MODEL_PATH = "/content/runs/detect/yolov9_hardhat/weights/best.pt"

st.title("🦺 Smart Safety Gear Detection System")
st.write("A real-time AI-powered system to verify if workers are wearing safety equipment like helmets and vests.")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: `{MODEL_PATH}`")
else:
    model = YOLO(MODEL_PATH)

    # Sidebar Info
    with st.sidebar:
        st.header("⚙️ Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.2, 1.0, 0.5)
        st.markdown("---")
        st.write("Developed by **Malav Joshi** 🚀")
        st.write("Powered by **YOLOv9** + **Streamlit**")

    # Upload Section
    st.markdown("### 📤 Upload an image or video")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
        temp_file.close()

        st.info("🔍 Analyzing... Please wait.")
        results = model.predict(source=temp_path, save=True, conf=conf_threshold)
        result = results[0]

        # Display Results
        output_dir = result.save_dir
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if file.endswith((".jpg", ".png")):
                st.image(file_path, caption="🖼️ Detection Output", use_column_width=True)
            elif file.endswith(".mp4"):
                st.video(file_path)

        # Analyze Detection Results
        detected_classes = [model.names[int(cls)] for cls in result.boxes.cls]
        detected_items = set([cls.lower() for cls in detected_classes])
        required_items = {"helmet", "vest"}

        st.markdown("### 🧾 Detection Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.success(f"✅ Detected Objects: {', '.join(detected_items) if detected_items else 'None'}")

        with col2:
            if required_items.issubset(detected_items):
                st.markdown("### 🟢 Status: SAFE")
                st.balloons()
                st.success("All required safety gear detected!")
            else:
                st.markdown("### 🔴 Status: UNSAFE")
                missing = required_items - detected_items
                st.error(f"Missing: {', '.join(missing)}")
                st.warning("Please ensure proper safety gear is worn before entering the site.")

    # Footer
    st.markdown('<div class="footer">© 2025 Smart Safety AI By Malav Joshi | YOLOv9 Detection Demo</div>', unsafe_allow_html=True)
