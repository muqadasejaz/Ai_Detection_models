# ─────────────────────────────────────────────────────────────────────────────
#  pages/2_🖼️_Image_Detection.py
#  AI Image Detector — EfficientNetV2-B3
#  Model loaded from Google Drive via st.secrets["effnet"]
# ─────────────────────────────────────────────────────────────────────────────
import os
import io
import numpy as np
import streamlit as st
from PIL import Image as PILImage

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

st.set_page_config(page_title="Image Detection · DeepSentinel", page_icon="🖼️", layout="wide")

from utils import inject_css, result_card_html
inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ DeepSentinel")
    st.caption("AI & Deepfake Detection Suite")

# ── Google Drive model downloader ─────────────────────────────────────────────
def _download_from_drive(file_id: str, dest_path: str) -> str:
    """
    Download a file from Google Drive by file_id.
    Uses the export/download URL — works for files shared with 'Anyone with link'.
    Returns dest_path on success, raises on failure.
    """
    import requests

    dest = dest_path
    if os.path.exists(dest):
        return dest  # already cached

    session = requests.Session()
    url = "https://drive.google.com/uc"
    params = {"id": file_id, "export": "download"}
    response = session.get(url, params=params, stream=True)

    # Handle the virus-scan warning page Google shows for large files
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        params["confirm"] = token
        response = session.get(url, params=params, stream=True)

    # Stream to disk
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

    return dest


@st.cache_resource(show_spinner="Loading model…")
def load_model(file_id: str):
    """
    Download cnn_detection.h5 from Drive (once, then cached),
    then load it with Keras.
    """
    import keras

    model_path = "/tmp/cnn_detection.h5"
    _download_from_drive(file_id, model_path)
    model = keras.models.load_model(model_path, compile=False)
    return model


def get_drive_file_id() -> str:
    """
    Pull the Google Drive file ID (or full URL).
    Checks st.secrets["gdrive"]["effnet"] first (nested section),
    then falls back to st.secrets["effnet"] (flat key).
    Accepts either:
      - A bare file ID:  "1AbCdEfGhIjKlMnOpQrStUvWxYz"
      - A full URL:      "https://drive.google.com/file/d/<id>/view?..."
    """
    if "gdrive" in st.secrets and "effnet" in st.secrets["gdrive"]:
        raw = st.secrets["gdrive"]["effnet"]
    elif "effnet" in st.secrets:
        raw = st.secrets["effnet"]
    else:
        raise KeyError("effnet")
    if "drive.google.com" in raw:
        # Extract the ID portion from the URL
        import re
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", raw)
        if match:
            return match.group(1)
        # Fallback: uc?id=... format
        match = re.search(r"id=([a-zA-Z0-9_-]+)", raw)
        if match:
            return match.group(1)
        raise ValueError(f"Could not parse Drive file ID from URL: {raw}")
    return raw.strip()


# ── Inference ─────────────────────────────────────────────────────────────────
def img_run(pil_img, model):
    """
    Preprocess and run a single PIL image through EfficientNetV2-B3.

    CRITICAL: Feed as float32 in [0, 255].
    DO NOT divide by 255 — the backbone uses include_preprocessing=True
    and normalises internally ([0,255] → [-1,1]).
    Input resolution: 224×224 (EfficientNetV2-B3 native).
    """
    img = pil_img.convert("RGB").resize((224, 224), PILImage.LANCZOS)
    arr = np.array(img, dtype=np.float32)          # [0, 255] — no /255
    arr = np.expand_dims(arr, axis=0)              # (1, 224, 224, 3)
    return model.predict(arr, verbose=0)


def img_interpret(preds):
    """
    score = P(AI-Generated).
    score >= 0.5  →  AI Generated
    score <  0.5  →  Real Photo
    """
    score = float(preds[0][0])
    if score >= 0.5:
        return "AI Generated", score, int(score * 100), "ai"
    else:
        return "Real Photo",   score, int((1 - score) * 100), "real"


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🖼️ AI Image Detector</div>', unsafe_allow_html=True)
st.caption("EfficientNetV2-B3 · 224×224 · Trained on 16 datasets")
st.divider()

if "img_prev" not in st.session_state:
    st.session_state.img_prev = None

il, ir = st.columns([1, 1], gap="large")

with il:
    uimg = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp"],
        key="iup",
    )

    if uimg != st.session_state.img_prev:
        st.session_state.img_prev = uimg

    if uimg:
        ib = uimg.read()
        if len(ib) / (1024 ** 2) > 20:
            st.error("Max 20 MB.")
            st.stop()
        st.image(ib, use_container_width=True)

with ir:
    rph = st.empty()

    if uimg:
        # Load model (downloads once, then cached)
        try:
            file_id = get_drive_file_id()
        except KeyError:
            st.error(
                '`effnet` secret not found. '
                'Add your Google Drive file ID under **Settings → Secrets** as `effnet`.'
            )
            st.stop()
        except ValueError as e:
            st.error(str(e))
            st.stop()

        try:
            model = load_model(file_id)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

        pil = PILImage.open(io.BytesIO(ib))

        with st.spinner("Running EfficientNetV2-B3…"):
            preds = img_run(pil, model)

        lbl, sc, cp, cls = img_interpret(preds)
        rph.markdown(
            result_card_html(lbl, cls, cp, sc, ""),
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="info-box">'
    '<b>EfficientNetV2-B3</b> · 224×224 · float32 [0, 255] (no /255) · '
    'Score ≥ 0.5 → AI Generated · Score &lt; 0.5 → Real Photo'
    '</div>',
    unsafe_allow_html=True,
)
