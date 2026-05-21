# ─────────────────────────────────────────────────────────────────────────────
#  pages/2_🖼️_Image_Detection.py
#  AI Image Detector — EfficientNetV2-B3
#
#  MODEL FACTS (from training notebook):
#    • Input  : 224 × 224 × 3,  float32 in [0, 255]  — DO NOT /255
#    • Backbone: EfficientNetV2-B3  include_preprocessing=True  (normalises internally)
#    • Output : sigmoid scalar   0 = Real Photo  ·  1 = AI Generated
#    • Decision: score >= 0.5 → AI Generated  ·  score < 0.5 → Real Photo
#    • Confidence displayed:
#        AI label  →  raw score  (how AI-generated it looks)
#        Real label→  1 - score  (how real it looks)
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ DeepSentinel")
    st.caption("AI & Deepfake Detection Suite")
    st.divider()
    st.markdown("### 📁 Model Path")
    effnet_v2_path = st.text_input(
        "EfficientNet Model (.h5)",
        value=os.environ.get("IMG_EFFNET_V2_PATH", "cnn_detection.h5"),
        help="Path to cnn_detection.h5 — EfficientNetV2-B3 trained on 16 datasets",
    )
    st.divider()
    st.markdown("### ⚙️ Detection Settings")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
        help=(
            "Score ≥ threshold → AI Generated · Score < threshold → Real Photo\n\n"
            "Lower this (e.g. 0.30) if the model misses AI images. "
            "Raise it (e.g. 0.60) to reduce false positives on real photos."
        ),
    )
    st.caption(f"Current: score ≥ **{threshold:.2f}** → AI Generated")

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading EfficientNet model…")
def load_effnet_v2(model_path: str):
    """
    Load saved EfficientNetV2-B3 model.
    Uses compile=False — we only need forward pass (no training).
    """
    import tensorflow as tf
    from tensorflow import keras
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: '{model_path}'\n"
            f"Set the correct path in the sidebar → Model Path."
        )
    model = keras.models.load_model(model_path, compile=False)
    # Verify expected shape
    assert model.input_shape[1:] == (224, 224, 3), (
        f"Unexpected input shape {model.input_shape}. Expected (None, 224, 224, 3)."
    )
    return model


def preprocess_image(pil_img: PILImage.Image) -> np.ndarray:
    """
    Exact replication of training's load_image() tf.function:
        img = tf.cast(img, tf.float32)  # [0, 255] — backbone normalises
    Steps:
      1. Convert to RGB
      2. Resize to 224×224 (LANCZOS for quality)
      3. Cast to float32  →  keep in [0, 255]
      4. Add batch dimension → (1, 224, 224, 3)

    ⚠  DO NOT divide by 255.
       EfficientNetV2-B3 was built with include_preprocessing=True,
       so the backbone rescales internally.
    """
    img = pil_img.convert("RGB")
    img = img.resize((224, 224), PILImage.LANCZOS)
    arr = np.array(img, dtype=np.float32)   # [0, 255]
    return np.expand_dims(arr, axis=0)      # (1, 224, 224, 3)


def run_inference(model, pil_img: PILImage.Image) -> float:
    """
    Run forward pass and return raw sigmoid score in [0, 1].
    score → 0 : model is confident it's a Real Photo
    score → 1 : model is confident it's AI Generated
    """
    arr = preprocess_image(pil_img)
    return float(model.predict(arr, verbose=0)[0][0])


def interpret_score(score: float, thresh: float) -> tuple:
    """
    Returns (label, css_class, confidence_pct, raw_score).

    Label logic matches training notebook exactly:
        score >= thresh  →  'AI Generated'
        score <  thresh  →  'Real Photo'

    Confidence shown to user:
        AI label  →  score          (how AI-like)
        Real label→  1 - score      (how real-like)
    """
    if score >= thresh:
        label      = "AI Generated"
        css_cls    = "ai"
        confidence = score              # high score = high AI confidence
    else:
        label      = "Real Photo"
        css_cls    = "real"
        confidence = 1.0 - score       # low score = high real confidence

    conf_pct = int(round(confidence * 100))
    return label, css_cls, conf_pct, score


# ── Confidence bar helper ─────────────────────────────────────────────────────
def confidence_bar_html(ai_score: float, thresh: float) -> str:
    """
    Dual progress bar showing AI probability vs Real probability.
    ai_score is the raw sigmoid output in [0, 1].
    """
    ai_pct   = int(round(ai_score * 100))
    real_pct = 100 - ai_pct
    thresh_pct = int(round(thresh * 100))

    return f"""
<div style="margin-top:1rem;">
  <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#3a3b5a;
              letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.5rem;">
    Score Breakdown
  </div>

  <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem;">
    <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#ff4060;
                min-width:4.5rem;">AI-Gen</div>
    <div style="flex:1;height:6px;background:#181928;border-radius:3px;overflow:hidden;">
      <div style="width:{ai_pct}%;height:100%;border-radius:3px;
                  background:linear-gradient(90deg,#ff3a5c,#ff8c00);"></div>
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:0.78rem;
                color:#d0d1e8;min-width:2.8rem;text-align:right;">{ai_pct}%</div>
  </div>

  <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.8rem;">
    <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#00d4aa;
                min-width:4.5rem;">Real</div>
    <div style="flex:1;height:6px;background:#181928;border-radius:3px;overflow:hidden;">
      <div style="width:{real_pct}%;height:100%;border-radius:3px;
                  background:linear-gradient(90deg,#00d4aa,#0090ff);"></div>
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:0.78rem;
                color:#d0d1e8;min-width:2.8rem;text-align:right;">{real_pct}%</div>
  </div>

  <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#2e2f4a;
              border-top:1px solid #181928;padding-top:0.5rem;line-height:1.8;">
    <b style="color:#3a3b5a;">Threshold</b> · {thresh:.2f}
    &nbsp;·&nbsp;
    <b style="color:#3a3b5a;">Raw score</b> · {ai_score:.6f}
    &nbsp;·&nbsp;
    <b style="color:#3a3b5a;">Decision</b> · score {'≥' if ai_score >= thresh else '<'} {thresh:.2f}
    → {'AI Generated' if ai_score >= thresh else 'Real Photo'}
  </div>
</div>
"""


def model_info_card_html() -> str:
    """Static card showing model architecture details."""
    return """
<div style="background:#0a0b18;border:1px solid #181928;border-radius:10px;
            padding:1rem 1.2rem;margin-top:1rem;">
  <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#3a3b5a;
              letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.6rem;">
    Model Architecture
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#44456a;line-height:1.9;">
    <b style="color:#5a5b7a;">Backbone</b> · EfficientNetV2-B3 (ImageNet pretrained)<br>
    <b style="color:#5a5b7a;">Head</b> · GAP → BN → Dense(512) → Dense(256) → Sigmoid<br>
    <b style="color:#5a5b7a;">Input</b> · 224 × 224 × 3 · float32 [0, 255]<br>
    <b style="color:#5a5b7a;">Preprocessing</b> · include_preprocessing=True (backbone normalises)<br>
    <b style="color:#5a5b7a;">Training</b> · 16 datasets · Phase-1 (5 ep) + Phase-2 fine-tune (5 ep)<br>
    <b style="color:#5a5b7a;">Labels</b> · 0 = Real Photo · 1 = AI Generated<br>
    <b style="color:#5a5b7a;">Default threshold</b> · 0.50
  </div>
</div>
"""


# ── Page UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🖼️ AI Image Detector</div>', unsafe_allow_html=True)
st.caption("EfficientNetV2-B3 · 16-dataset training · 224×224 input")
st.divider()

# Session state init
if "img_prev" not in st.session_state:
    st.session_state.img_prev = None
if "img_key" not in st.session_state:
    st.session_state.img_key = "imgkey_0"

col_left, col_right = st.columns([1, 1], gap="large")

# ── Left column: upload ───────────────────────────────────────────────────────
with col_left:
    uploaded = st.file_uploader(
        "Upload an image to analyse",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        key="img_uploader",
        help="Supported: PNG, JPG, JPEG, WEBP, BMP · Max 20 MB",
    )

    # Reset key on new file so selectbox resets
    if uploaded != st.session_state.img_prev:
        st.session_state.img_key = (
            "imgkey_1" if st.session_state.img_key == "imgkey_0" else "imgkey_0"
        )
        st.session_state.img_prev = uploaded

    if uploaded:
        raw_bytes = uploaded.read()
        if len(raw_bytes) / (1024 ** 2) > 20:
            st.error("❌  File too large. Maximum size is 20 MB.")
            st.stop()

        # Show uploaded image
        st.image(raw_bytes, use_container_width=True)

        # File metadata
        file_size_kb = len(raw_bytes) / 1024
        try:
            pil_preview  = PILImage.open(io.BytesIO(raw_bytes))
            orig_w, orig_h = pil_preview.size
            st.caption(
                f"📐 {orig_w} × {orig_h} px  ·  "
                f"📁 {file_size_kb:.1f} KB  ·  "
                f"🎨 {pil_preview.mode}"
            )
        except Exception:
            pass

# ── Right column: result ──────────────────────────────────────────────────────
with col_right:
    result_placeholder = st.empty()

    if uploaded:
        # Load model (cached)
        try:
            model = load_effnet_v2(effnet_v2_path)
        except FileNotFoundError as e:
            result_placeholder.error(f"⚠️  {e}")
            st.stop()
        except AssertionError as e:
            result_placeholder.error(f"⚠️  Model shape mismatch: {e}")
            st.stop()
        except Exception as e:
            result_placeholder.error(f"⚠️  Failed to load model: {e}")
            st.stop()

        # Run inference
        with st.spinner("Running EfficientNet model…"):
            try:
                pil_img = PILImage.open(io.BytesIO(raw_bytes))
                score   = run_inference(model, pil_img)
            except Exception as e:
                result_placeholder.error(f"⚠️  Inference failed: {e}")
                st.stop()

        # Interpret result
        label, css_cls, conf_pct, raw_score = interpret_score(score, threshold)

        # ── Result card ───────────────────────────────────────────────────────
        meta_html = (
            f"<b>Model</b> · EfficientNet  "
            f"&nbsp;·&nbsp; "
            f"<b>File</b> · {uploaded.name}"
        )
        result_placeholder.markdown(
            result_card_html(label, css_cls, conf_pct, raw_score, meta_html),
            unsafe_allow_html=True,
        )

        # ── Dual confidence bars ──────────────────────────────────────────────
        st.markdown(
            confidence_bar_html(raw_score, threshold),
            unsafe_allow_html=True,
        )

        # ── Resized preview (what the model actually saw) ─────────────────────
        with st.expander("🔍 Show model input (224×224 resize)"):
            resized = pil_img.convert("RGB").resize((224, 224), PILImage.LANCZOS)
            st.image(resized, caption="Exact pixels fed to model (224×224)", width=224)
            st.caption(
                "⚠️ Preprocessing: float32 [0, 255] — NOT divided by 255. "
                "EfficientNetV2-B3 normalises internally via include_preprocessing=True."
            )

    else:
        # Idle state
        result_placeholder.markdown(
            """
<div style="background:#0e0f1a;border:1px dashed #1c1d30;border-radius:14px;
            padding:2.5rem 2rem;text-align:center;margin-top:1rem;">
  <div style="font-size:2.5rem;margin-bottom:0.8rem;">🖼️</div>
  <div style="font-family:'DM Sans',sans-serif;font-weight:600;font-size:1rem;
              color:#44456a;margin-bottom:0.4rem;">
    Upload an image to detect
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#2e2f4a;
              line-height:1.7;">
    PNG · JPG · JPEG · WEBP · BMP<br>
    Real photographs vs AI-generated images
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

# ── Model info footer ─────────────────────────────────────────────────────────
st.divider()
st.markdown(model_info_card_html(), unsafe_allow_html=True)

# ── Bottom info box (matches app.py style) ────────────────────────────────────
st.markdown(
    """
<div class="info-box">
  <b>EfficientNet Model</b> · EfficientNetV2-B3 · 224×224 · float32 [0,255] — no /255 division<br>
  <b>Training</b> · 16 datasets (CIFAKE, ArtiFact, MidJourney, DALL-E, Deepfake-2026 & more)
  · Phase-1 head training (5 ep) + Phase-2 fine-tune layers 200+ (5 ep)<br>
  <b>Decision</b> · Score &lt; threshold → Real Photo · Score ≥ threshold → AI Generated<br>
  <b>Tip</b> · Adjust the threshold slider in the sidebar if the model misses AI images
  (lower it to ~0.30) or produces false alarms on real photos (raise it to ~0.60)
</div>
""",
    unsafe_allow_html=True,
)
