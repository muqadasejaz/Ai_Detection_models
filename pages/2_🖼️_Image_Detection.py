# ─────────────────────────────────────────────────────────────────────────────
#  pages/2_🖼️_Image_Detection.py
#  AI Image Detector — EfficientNetV2-B3
#
#  Training facts (ai-vs-real-images.ipynb):
#    Input  : 224x224x3, float32 [0,255] — DO NOT /255
#             include_preprocessing=True normalises inside backbone
#    Output : sigmoid scalar — 0=Real, 1=AI Generated
#    Rule   : score >= 0.5 -> AI Generated | score < 0.5 -> Real Photo
#
#  Secret:
#    [gdrive]
#    cnn_detection = "GOOGLE_DRIVE_FILE_ID"
# ─────────────────────────────────────────────────────────────────────────────
import os
import io

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import streamlit as st
from PIL import Image as PILImage

st.set_page_config(
    page_title="Image Detection · DeepSentinel",
    page_icon="🖼️",
    layout="wide",
)

from utils import inject_css, result_card_html, ensure_file, load_keras_model
inject_css()

IMG_SIZE       = (224, 224)
# FIX: must match the key in utils.DRIVE_FILES ("cnn_detection.h5") so the file
# is downloaded exactly once instead of once here + once by the old downloader.
MODEL_FILENAME = "cnn_detection.h5"


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL — downloaded once (via ensure_file), loaded once, reused forever.
#  @st.cache_resource keeps the model object in memory for the app lifetime;
#  every page visit and every upload reuses the same object.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading EfficientNet model... (first visit only)")
def get_model():
    path = ensure_file(MODEL_FILENAME)          # downloads once if missing
    # Keras 3 first (this model was saved with Keras 3 → uses `batch_shape`),
    # falling back to legacy Keras 2 if needed.
    return load_keras_model(path, compile=False)


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(pil_img: PILImage.Image) -> np.ndarray:
    """
    Matches training load_image() exactly:
      resize to 224x224, cast float32, keep [0,255].
    DO NOT divide by 255 — backbone uses include_preprocessing=True.
    """
    img = pil_img.convert("RGB").resize(IMG_SIZE, PILImage.LANCZOS)
    arr = np.array(img, dtype=np.float32)   # [0.0 .. 255.0]
    return np.expand_dims(arr, axis=0)      # (1, 224, 224, 3)


def predict(model, pil_img: PILImage.Image) -> float:
    return float(model.predict(preprocess(pil_img), verbose=0)[0][0])


def interpret(score: float):
    if score >= 0.5:
        return "AI Generated", "ai",   int(round(score * 100))
    return "Real Photo",       "real", int(round((1.0 - score) * 100))


# ─────────────────────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ DeepSentinel")
    st.caption("AI & Deepfake Detection Suite")

st.markdown('<div class="hero-title">🖼️ AI Image Detector</div>', unsafe_allow_html=True)
st.divider()

# Load model once — cached across all sessions and all image uploads
model = get_model()

if "img_prev" not in st.session_state:
    st.session_state.img_prev = None

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    uploaded = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        key="img_uploader",
    )
    if uploaded != st.session_state.img_prev:
        st.session_state.img_prev = uploaded
    raw = None
    if uploaded:
        raw = uploaded.read()
        if len(raw) / (1024 ** 2) > 20:
            st.error("Max 20 MB.")
            st.stop()
        st.image(raw, use_container_width=True)

with col_right:
    result_slot = st.empty()

    if uploaded and raw is not None:
        with st.spinner("Analysing image..."):
            try:
                pil   = PILImage.open(io.BytesIO(raw))
                score = predict(model, pil)
            except Exception as exc:
                result_slot.error(f"Inference failed: {exc}")
                st.stop()

        label, css_cls, conf_pct = interpret(score)
        result_slot.markdown(
            result_card_html(
                label, css_cls, conf_pct, score,
                f"<b>Model</b> · EfficientNet &nbsp;·&nbsp; <b>File</b> · {uploaded.name}",
            ),
            unsafe_allow_html=True,
        )
