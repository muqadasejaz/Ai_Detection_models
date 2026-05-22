import os
import io
import numpy as np
import streamlit as st
from PIL import Image as PILImage

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
st.set_page_config(page_title="Image Detection · DeepSentinel", page_icon="🖼️", layout="wide")

from utils import inject_css, result_card_html
inject_css()

IMG_SIZE = (224, 224)
MODEL_FILENAME = "cnn_detection.h5"

@st.cache_resource(show_spinner="Loading EfficientNet model...")
def get_model():
    if not os.path.exists(MODEL_FILENAME) or os.path.getsize(MODEL_FILENAME) < 10_000_000:
        st.error("Model not found. Run from main page first.")
        st.stop()
    from tensorflow import keras
    return keras.models.load_model(MODEL_FILENAME, compile=False)

def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE, PILImage.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def predict(model, pil_img):
    return float(model.predict(preprocess(pil_img), verbose=0)[0][0])

def interpret(score):
    if score >= 0.5:
        return "AI Generated", "ai", int(round(score * 100))
    return "Real Photo", "real", int(round((1 - score) * 100))

with st.sidebar:
    st.markdown("## 🛡️ DeepSentinel")

st.markdown('<div class="hero-title">🖼️ AI Image Detector</div>', unsafe_allow_html=True)
st.divider()

uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp","bmp"])

if uploaded:
    raw = uploaded.read()
    if len(raw) > 20 * 1024 * 1024:
        st.error("Maximum 20 MB allowed.")
        st.stop()
    st.image(raw, use_container_width=True)

    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                model = get_model()
                pil = PILImage.open(io.BytesIO(raw))
                score = predict(model, pil)
                label, css_cls, conf = interpret(score)
                st.markdown(result_card_html(label, css_cls, conf, score, f"File: {uploaded.name}"), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
