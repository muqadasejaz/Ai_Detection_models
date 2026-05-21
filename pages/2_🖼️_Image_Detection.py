# ─────────────────────────────────────────────────────────────────────────────
#  pages/2_🖼️_Image_Detection.py
#  AI Image Detector — EfficientNetV2-B3
#
#  Training notebook facts (ai-vs-real-images.ipynb):
#    Architecture : EfficientNetV2-B3 (ImageNet) + GAP + BN +
#                   Dense(512,relu) + Dropout(0.4) +
#                   Dense(256,relu) + Dropout(0.3) + Dense(1,sigmoid)
#    Input size   : 224 x 224 x 3
#    Preprocessing: float32 in [0, 255]  — DO NOT divide by 255
#                   include_preprocessing=True handles normalisation internally
#    Labels       : 0 = Real Photo  |  1 = AI Generated
#    Decision     : score >= 0.5 -> AI Generated
#                   score <  0.5 -> Real Photo
#    Saved as     : cnn_detection.h5  (full model)
#
#  Secrets (Streamlit Cloud -> App Settings -> Secrets):
#    [gdrive]
#    cnn_detection = "YOUR_GOOGLE_DRIVE_FILE_ID"
# ─────────────────────────────────────────────────────────────────────────────

import os
import io
import numpy as np
import streamlit as st
from PIL import Image as PILImage

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

st.set_page_config(
    page_title="Image Detection · DeepSentinel",
    page_icon="🖼️",
    layout="wide",
)

from utils import inject_css, result_card_html
inject_css()

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_FILENAME = "cnn_detection.h5"
IMG_SIZE = (224, 224)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_file_id() -> str:
    gdrive = st.secrets.get("gdrive", {})
    fid = gdrive.get("cnn_detection", "").strip()
    if not fid:
        fid = st.secrets.get("cnn_detection", "").strip()
    return fid


# ── Download (once, cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _ensure_model_on_disk() -> str:
    if os.path.exists(MODEL_FILENAME) and os.path.getsize(MODEL_FILENAME) > 1_000_000:
        return MODEL_FILENAME

    file_id = _get_file_id()
    if not file_id:
        st.error(
            "Model file not found and no Drive secret configured.\n\n"
            "Add this to Streamlit Secrets:\n\n"
            "```toml\n[gdrive]\ncnn_detection = \"YOUR_FILE_ID\"\n```"
        )
        st.stop()

    try:
        import gdown
    except ImportError:
        st.error("`gdown` not installed — add `gdown>=5.1.0` to requirements.txt")
        st.stop()

    progress = st.progress(0, text="Downloading EfficientNet model from Google Drive...")
    try:
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            MODEL_FILENAME,
            quiet=False,
        )
        progress.progress(1.0, text="Download complete!")
        progress.empty()
    except Exception as exc:
        progress.empty()
        st.error(
            f"Download failed: {exc}\n\n"
            "Check the file ID is correct and the file is shared as "
            "'Anyone with the link can view'."
        )
        st.stop()

    if not os.path.exists(MODEL_FILENAME) or os.path.getsize(MODEL_FILENAME) < 1_000_000:
        st.error("Downloaded file looks corrupt (too small). Re-check the Drive link.")
        st.stop()

    return MODEL_FILENAME


# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading EfficientNet model...")
def _load_model(model_path: str):
    from tensorflow import keras
    model = keras.models.load_model(model_path, compile=False)
    return model


# ── Preprocessing — must match training exactly ────────────────────────────────
def _preprocess(pil_img: PILImage.Image) -> np.ndarray:
    # Training: tf.image.resize -> tf.cast(float32)  — NO /255
    # include_preprocessing=True normalises [0,255] -> [-1,1] inside the backbone
    img = pil_img.convert("RGB").resize(IMG_SIZE, PILImage.LANCZOS)
    arr = np.array(img, dtype=np.float32)   # [0.0, 255.0]
    return np.expand_dims(arr, axis=0)      # (1, 224, 224, 3)


def _predict(model, pil_img: PILImage.Image) -> float:
    return float(model.predict(_preprocess(pil_img), verbose=0)[0][0])


def _interpret(score: float):
    # score = P(AI-Generated)
    if score >= 0.5:
        label, css_cls = "AI Generated", "ai"
        conf_pct = int(round(score * 100))
    else:
        label, css_cls = "Real Photo", "real"
        conf_pct = int(round((1.0 - score) * 100))
    return label, css_cls, conf_pct


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ DeepSentinel")
    st.caption("AI & Deepfake Detection Suite")

# ── Page ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🖼️ AI Image Detector</div>', unsafe_allow_html=True)
st.divider()

# Ensure model is ready before the uploader appears
model_path = _ensure_model_on_disk()

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
    if uploaded:
        raw = uploaded.read()
        if len(raw) / (1024 ** 2) > 20:
            st.error("Max 20 MB.")
            st.stop()
        st.image(raw, use_container_width=True)

with col_right:
    result_slot = st.empty()

    if uploaded:
        try:
            model = _load_model(model_path)
        except Exception as exc:
            result_slot.error(f"Failed to load model: {exc}")
            st.stop()

        with st.spinner("Running EfficientNet model..."):
            try:
                pil = PILImage.open(io.BytesIO(raw))
                score = _predict(model, pil)
            except Exception as exc:
                result_slot.error(f"Inference failed: {exc}")
                st.stop()

        label, css_cls, conf_pct = _interpret(score)
        result_slot.markdown(
            result_card_html(
                label, css_cls, conf_pct, score,
                f"<b>Model</b> · EfficientNet &nbsp;·&nbsp; <b>File</b> · {uploaded.name}",
            ),
            unsafe_allow_html=True,
        )
