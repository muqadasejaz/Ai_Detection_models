# ─────────────────────────────────────────────────────────────────────────────
#  pages/2_🖼️_Image_Detection.py
#  AI Image Detector — EfficientNetV2-B3
#
#  MODEL FACTS (from training notebook ai-vs-real-images.ipynb):
#    • Architecture : EfficientNetV2-B3 + GAP → BN → Dense(512) → Dense(256) → Sigmoid
#    • Input        : 224 × 224 × 3,  float32 in [0, 255]  ← DO NOT /255
#    • Preprocessing: include_preprocessing=True  (backbone normalises internally)
#    • Output       : sigmoid scalar   0 = Real Photo  ·  1 = AI Generated
#    • Decision     : score >= threshold → AI Generated  ·  score < threshold → Real Photo
#
#  GOOGLE DRIVE SETUP  (Streamlit Cloud → App Settings → Secrets):
#
#    [gdrive]
#    cnn_detection = "YOUR_GOOGLE_DRIVE_FILE_ID"
#
#    File ID is in the Drive share URL:
#    https://drive.google.com/file/d/  >>>FILE_ID<<<  /view
#    File must be shared as "Anyone with the link (Viewer)".
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

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_FILENAME = "cnn_detection.h5"
IMG_SIZE       = (224, 224)


# ─────────────────────────────────────────────────────────────────────────────
#  GOOGLE DRIVE DOWNLOADER
#  Downloads once, stores to disk.  Subsequent runs reuse the cached file.
#  File ID comes from  st.secrets["gdrive"]["cnn_detection"]  — never hardcoded.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def download_image_model() -> str:
    """
    Ensures MODEL_FILENAME exists locally.  Returns local file path.

    Priority:
      1. Already on disk        → skip download, return path.
      2. Secret present         → gdown from Drive, return path.
      3. No secret, not on disk → show clear instructions, return path (load fails gracefully).
    """
    if os.path.exists(MODEL_FILENAME):
        return MODEL_FILENAME

    gdrive_secrets = st.secrets.get("gdrive", {})
    file_id        = gdrive_secrets.get("cnn_detection", "").strip()

    if not file_id:
        st.warning(
            "⚠️  **Model not on disk and no Drive secret found.**\n\n"
            "Add this to **Streamlit Cloud → App Settings → Secrets**:\n\n"
            "```toml\n[gdrive]\ncnn_detection = \"YOUR_GOOGLE_DRIVE_FILE_ID\"\n```\n\n"
            "Find the file ID in your Drive share link:\n"
            "`https://drive.google.com/file/d/`**`FILE_ID`**`/view`"
        )
        return MODEL_FILENAME  # load_effnet_v2 will raise FileNotFoundError gracefully

    try:
        import gdown
    except ImportError:
        st.error("❌  `gdown` not installed. Add `gdown` to `requirements.txt` and redeploy.")
        st.stop()

    bar = st.progress(0, text="⬇️  Downloading EfficientNet model from Google Drive…")
    try:
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            MODEL_FILENAME,
            quiet=False,
        )
        bar.progress(1.0, text="✅  Model downloaded!")
        bar.empty()
    except Exception as e:
        bar.empty()
        st.error(
            f"❌  Download failed: {e}\n\n"
            "Check:\n"
            "1. File ID in secrets is correct.\n"
            "2. File is shared as **Anyone with the link → Viewer**.\n"
            "3. File is under ~1 GB (gdown limit without chunked download)."
        )
        st.stop()

    return MODEL_FILENAME


# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading EfficientNet model…")
def load_effnet_v2(model_path: str):
    """Load EfficientNetV2-B3. compile=False — inference only."""
    from tensorflow import keras

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. "
            "Ensure the Drive secret is set and the app was restarted."
        )

    model = keras.models.load_model(model_path, compile=False)

    if model.input_shape[1:] != (224, 224, 3):
        raise ValueError(
            f"Unexpected input shape {model.input_shape[1:]} — expected (224, 224, 3). "
            "Is this the correct cnn_detection.h5?"
        )
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(pil_img: PILImage.Image) -> np.ndarray:
    """
    Matches training notebook's load_image() exactly:
        img = tf.image.resize(img, IMG_SIZE, method='bilinear')
        img = tf.cast(img, tf.float32)   # [0, 255] — backbone normalises

    CRITICAL: do NOT divide by 255.
    EfficientNetV2-B3 uses include_preprocessing=True which rescales
    [0,255]→[-1,1] inside the backbone.  Dividing first would break the model.
    """
    img = pil_img.convert("RGB")
    img = img.resize(IMG_SIZE, PILImage.LANCZOS)
    arr = np.array(img, dtype=np.float32)   # [0.0, 255.0]  ← NO /255
    return np.expand_dims(arr, axis=0)      # (1, 224, 224, 3)


def run_inference(model, pil_img: PILImage.Image) -> float:
    """Forward pass. Returns sigmoid score ∈ [0, 1]."""
    return float(model.predict(preprocess_image(pil_img), verbose=0)[0][0])


def interpret_score(score: float, thresh: float) -> tuple:
    """
    score >= thresh  →  'AI Generated'   css='ai'    conf = score
    score <  thresh  →  'Real Photo'     css='real'  conf = 1 - score
    Returns (label, css_cls, conf_pct, raw_score)
    """
    if score >= thresh:
        return "AI Generated", "ai",   int(round(score * 100)),         score
    else:
        return "Real Photo",   "real", int(round((1.0 - score) * 100)), score


# ── HTML helpers ──────────────────────────────────────────────────────────────
def score_breakdown_html(score: float, thresh: float) -> str:
    ai_pct   = int(round(score * 100))
    real_pct = 100 - ai_pct
    decision = "AI Generated" if score >= thresh else "Real Photo"
    op       = "≥" if score >= thresh else "<"
    return f"""
<div style="margin-top:1rem;">
  <div style="font-family:'DM Mono',monospace;font-size:.62rem;color:#3a3b5a;
              letter-spacing:.14em;text-transform:uppercase;margin-bottom:.5rem;">
    Score Breakdown
  </div>
  <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.4rem;">
    <div style="font-family:'DM Mono',monospace;font-size:.72rem;color:#ff4060;
                min-width:4.5rem;">AI-Gen</div>
    <div style="flex:1;height:6px;background:#181928;border-radius:3px;overflow:hidden;">
      <div style="width:{ai_pct}%;height:100%;border-radius:3px;
                  background:linear-gradient(90deg,#ff3a5c,#ff8c00);"></div>
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:.78rem;
                color:#d0d1e8;min-width:2.8rem;text-align:right;">{ai_pct}%</div>
  </div>
  <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.8rem;">
    <div style="font-family:'DM Mono',monospace;font-size:.72rem;color:#00d4aa;
                min-width:4.5rem;">Real</div>
    <div style="flex:1;height:6px;background:#181928;border-radius:3px;overflow:hidden;">
      <div style="width:{real_pct}%;height:100%;border-radius:3px;
                  background:linear-gradient(90deg,#00d4aa,#0090ff);"></div>
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:.78rem;
                color:#d0d1e8;min-width:2.8rem;text-align:right;">{real_pct}%</div>
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:.6rem;color:#2e2f4a;
              border-top:1px solid #181928;padding-top:.5rem;line-height:1.8;">
    <b style="color:#3a3b5a;">Raw score</b> · {score:.6f}
    &nbsp;·&nbsp;
    <b style="color:#3a3b5a;">Threshold</b> · {thresh:.2f}
    &nbsp;·&nbsp;
    <b style="color:#3a3b5a;">Decision</b> · {score:.4f} {op} {thresh:.2f} → {decision}
  </div>
</div>"""


def model_info_card_html(on_disk: bool) -> str:
    status_c = "#00d4aa" if on_disk else "#ff4060"
    status_t = "Ready ✓" if on_disk else "Not loaded ✗"
    return f"""
<div style="background:#0a0b18;border:1px solid #181928;border-radius:10px;
            padding:1rem 1.2rem;margin-top:1rem;">
  <div style="font-family:'DM Mono',monospace;font-size:.62rem;color:#3a3b5a;
              letter-spacing:.14em;text-transform:uppercase;margin-bottom:.6rem;">
    Model Details
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:.65rem;color:#44456a;line-height:2;">
    <b style="color:#5a5b7a;">Name</b> · EfficientNet Model
    &nbsp;·&nbsp;
    <b style="color:{status_c};">{status_t}</b><br>
    <b style="color:#5a5b7a;">Architecture</b> · EfficientNetV2-B3 + GAP → Dense(512) → Dense(256) → Sigmoid<br>
    <b style="color:#5a5b7a;">Input</b> · 224 × 224 × 3 · float32 [0, 255] · no /255 division<br>
    <b style="color:#5a5b7a;">Preprocessing</b> · include_preprocessing=True (backbone normalises [0,255]→[-1,1])<br>
    <b style="color:#5a5b7a;">Training</b> · 16 datasets · Phase-1 head only (5 ep) + Phase-2 fine-tune (5 ep)<br>
    <b style="color:#5a5b7a;">Labels</b> · 0 = Real Photo · 1 = AI Generated<br>
    <b style="color:#5a5b7a;">Weights</b> · {MODEL_FILENAME} via st.secrets["gdrive"]["cnn_detection"]
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ DeepSentinel")
    st.caption("AI & Deepfake Detection Suite")
    st.divider()

    st.markdown("### ⚙️ Detection Settings")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10, max_value=0.90, value=0.50, step=0.01,
        help=(
            "score ≥ threshold  →  AI Generated\n"
            "score <  threshold →  Real Photo\n\n"
            "Lower to ~0.30 if the model misses AI images.\n"
            "Raise to ~0.60 to reduce false alarms on real photos."
        ),
    )
    st.caption(f"score ≥ **{threshold:.2f}** → AI Generated")

    st.divider()
    st.markdown("### ☁️ Model Status")
    disk_ok   = os.path.exists(MODEL_FILENAME)
    secret_ok = bool(st.secrets.get("gdrive", {}).get("cnn_detection", "").strip())

    if disk_ok:
        st.success(f"✅  `{MODEL_FILENAME}` on disk")
    elif secret_ok:
        st.info("☁️  Drive secret found — downloads on first run")
    else:
        st.error("❌  Drive secret missing")
        with st.expander("Setup instructions"):
            st.markdown(
                "**Streamlit Cloud → App Settings → Secrets:**\n\n"
                "```toml\n[gdrive]\ncnn_detection = \"YOUR_FILE_ID\"\n```\n\n"
                "**Get your file ID:**\n"
                "Share the file → copy link → ID is between `/d/` and `/view`\n\n"
                "**Share settings:**\n"
                "Anyone with the link → Viewer"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  DOWNLOAD ON PAGE LOAD  (runs once, cached)
# ─────────────────────────────────────────────────────────────────────────────
model_path    = download_image_model()
model_on_disk = os.path.exists(model_path)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🖼️ AI Image Detector</div>', unsafe_allow_html=True)
st.caption("EfficientNetV2-B3 · 16-dataset training · 224 × 224")
st.divider()

# ── session state for uploader key reset ──────────────────────────────────────
if "img_prev" not in st.session_state: st.session_state.img_prev = None
if "img_key"  not in st.session_state: st.session_state.img_key  = "imgk0"

col_left, col_right = st.columns([1, 1], gap="large")

# ── Left: Upload ──────────────────────────────────────────────────────────────
with col_left:
    uploaded = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        key="img_uploader",
    )
    if uploaded != st.session_state.img_prev:
        st.session_state.img_key  = "imgk1" if st.session_state.img_key == "imgk0" else "imgk0"
        st.session_state.img_prev = uploaded

    if uploaded:
        raw_bytes = uploaded.read()
        if len(raw_bytes) / (1024 ** 2) > 20:
            st.error("❌  Max 20 MB."); st.stop()
        st.image(raw_bytes, use_container_width=True)
        try:
            pil_info       = PILImage.open(io.BytesIO(raw_bytes))
            orig_w, orig_h = pil_info.size
            st.caption(
                f"📐 {orig_w} × {orig_h} px  ·  "
                f"📁 {len(raw_bytes)/1024:.1f} KB  ·  "
                f"🎨 {pil_info.mode}"
            )
        except Exception:
            pass

# ── Right: Result ─────────────────────────────────────────────────────────────
with col_right:
    result_ph = st.empty()

    if uploaded:
        # Load model
        try:
            model = load_effnet_v2(model_path)
        except FileNotFoundError as e:
            result_ph.error(f"⚠️  {e}"); st.stop()
        except ValueError as e:
            result_ph.error(f"⚠️  {e}"); st.stop()
        except Exception as e:
            result_ph.error(f"⚠️  Failed to load model: {e}"); st.stop()

        # Inference
        with st.spinner("Running EfficientNet model…"):
            try:
                pil_img = PILImage.open(io.BytesIO(raw_bytes))
                score   = run_inference(model, pil_img)
            except Exception as e:
                result_ph.error(f"⚠️  Inference failed: {e}"); st.stop()

        # Display result
        label, css_cls, conf_pct, raw_score = interpret_score(score, threshold)

        result_ph.markdown(
            result_card_html(
                label, css_cls, conf_pct, raw_score,
                f"<b>Model</b> · EfficientNet &nbsp;·&nbsp; <b>File</b> · {uploaded.name}",
            ),
            unsafe_allow_html=True,
        )

        # Score breakdown bars
        st.markdown(score_breakdown_html(raw_score, threshold), unsafe_allow_html=True)

        # Model input preview
        with st.expander("🔍 Model input preview (224 × 224)"):
            resized = pil_img.convert("RGB").resize(IMG_SIZE, PILImage.LANCZOS)
            st.image(resized, caption="Exact pixels fed to EfficientNetV2-B3", width=224)
            st.caption(
                "float32 [0, 255] — no /255 division. "
                "include_preprocessing=True normalises inside the backbone."
            )

    else:
        result_ph.markdown(
            """
<div style="background:#0e0f1a;border:1px dashed #1c1d30;border-radius:14px;
            padding:2.5rem 2rem;text-align:center;margin-top:1rem;">
  <div style="font-size:2.5rem;margin-bottom:.8rem;">🖼️</div>
  <div style="font-family:'DM Sans',sans-serif;font-weight:600;
              font-size:1rem;color:#44456a;margin-bottom:.4rem;">
    Upload an image to detect
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:.65rem;
              color:#2e2f4a;line-height:1.7;">
    PNG · JPG · JPEG · WEBP · BMP · max 20 MB<br>
    Real photographs vs AI-generated images
  </div>
</div>""",
            unsafe_allow_html=True,
        )

# ── Model info card ───────────────────────────────────────────────────────────
st.divider()
st.markdown(model_info_card_html(model_on_disk), unsafe_allow_html=True)

</div>""",
    unsafe_allow_html=True,
)
