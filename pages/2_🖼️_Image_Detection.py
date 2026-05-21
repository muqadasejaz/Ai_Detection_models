# ─────────────────────────────────────────────────────────────────────────────
#  pages/2_🖼️_Image_Detection.py
#  AI Image Detector — EfficientNetV2-B3
#  Model loaded from Google Drive via st.secrets["gdrive"]["effnet"]
# ─────────────────────────────────────────────────────────────────────────────
import os
import io
import sys
import subprocess
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
    Download a file from Google Drive using gdown.
    gdown handles the large-file virus-scan confirmation page that
    raw requests calls miss, which causes Drive to return an HTML page
    instead of the actual .h5 file (producing the 'file signature not found' error).
    """
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
        return dest_path  # already cached and looks valid

    # Install gdown if not present
    try:
        import gdown
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "gdown>=4.0.0"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import gdown

    gdown.download(id=file_id, output=dest_path, quiet=True)

    if not os.path.exists(dest_path) or os.path.getsize(dest_path) < 1_000_000:
        raise RuntimeError(
            "Model download failed or file is too small. "
            "Make sure the Drive file is shared as 'Anyone with the link can view'."
        )

    return dest_path


@st.cache_resource(show_spinner="Downloading and loading model…")
def load_model(file_id: str):
    import keras
    model_path = "/tmp/cnn_detection.h5"
    _download_from_drive(file_id, model_path)
    return keras.models.load_model(model_path, compile=False)


def get_drive_file_id() -> str:
    """
    Read the Drive file ID from secrets.
    Supports both layouts:
      [gdrive]
      effnet = "1t_X..."      ← nested (what the user has)

      effnet = "1t_X..."      ← flat fallback
    Also accepts a full Drive URL instead of a bare ID.
    """
    if "gdrive" in st.secrets and "effnet" in st.secrets["gdrive"]:
        raw = st.secrets["gdrive"]["effnet"]
    elif "effnet" in st.secrets:
        raw = st.secrets["effnet"]
    else:
        raise KeyError("effnet")

    raw = raw.strip()

    if "drive.google.com" in raw:
        import re
        m = re.search(r"/d/([a-zA-Z0-9_-]+)", raw) or \
            re.search(r"id=([a-zA-Z0-9_-]+)", raw)
        if not m:
            raise ValueError(f"Could not parse Drive file ID from: {raw}")
        return m.group(1)

    return raw


# ── Inference ─────────────────────────────────────────────────────────────────
def img_run(pil_img, model):
    """
    EfficientNetV2-B3 preprocessing:
      - Resize to 224×224 (native resolution)
      - Cast to float32, keep range [0, 255]
      - DO NOT divide by 255 — backbone uses include_preprocessing=True
        and normalises internally
    """
    img = pil_img.convert("RGB").resize((224, 224), PILImage.LANCZOS)
    arr = np.array(img, dtype=np.float32)   # [0, 255] — no /255
    arr = np.expand_dims(arr, axis=0)       # (1, 224, 224, 3)
    return model.predict(arr, verbose=0)


def img_interpret(preds):
    """
    score = P(AI-Generated)
    >= 0.5  →  AI Generated
    <  0.5  →  Real Photo
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
        try:
            file_id = get_drive_file_id()
        except KeyError:
            st.error(
                "`effnet` secret not found. Add it under **Settings → Secrets**:\n\n"
                "```toml\n[gdrive]\neffnet = \"your_drive_file_id\"\n```"
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
    "<b>EfficientNetV2-B3</b> · 224×224 · float32 [0, 255] (no /255) · "
    "Score ≥ 0.5 → AI Generated · Score &lt; 0.5 → Real Photo"
    "</div>",
    unsafe_allow_html=True,
)
