# ─────────────────────────────────────────────────────────────────────────────
#  pages/2_🖼️_Image_Detection.py
#  AI Image Detector — CNN · EfficientNetB3 · EfficientNet Art
#  Only tensorflow/keras loads when this page is visited.
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
    st.divider()
    st.markdown("### 📁 Model Paths")
    img_effnet_path     = st.text_input("EfficientNetB3 (.h5)",   value=os.environ.get("IMG_EFFNET_PATH",     "efficientnetb3_binary_classifier_8.h5"))
    img_effnet_art_path = st.text_input("EfficientNet Art (.h5)", value=os.environ.get("IMG_EFFNET_ART_PATH", "EfficientNet_fine_tune_art_model.h5"))
    img_cnn_weights     = st.text_input("CNN weights (.h5)",      value=os.environ.get("IMG_CNN_WEIGHTS",     "model_weights.weights.h5"))

# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading image models…")
def load_image_models(ep, ap, cp):
    import keras
    import tensorflow as tf
    missing = [p for p in [ep, ap, cp] if not os.path.exists(p)]
    if missing: raise FileNotFoundError(f"Missing: {missing}")
    eff     = keras.models.load_model(ep, compile=False)
    eff_art = keras.models.load_model(ap, compile=False)
    return eff, eff_art, cp

def _strategy():
    import tensorflow as tf
    try:
        res = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(res)
        tf.tpu.experimental.initialize_tpu_system(res)
        return tf.distribute.TPUStrategy(res)
    except ValueError:
        return tf.distribute.get_strategy()

def build_cnn(weights_path):
    from keras.models import Sequential
    from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
    m = Sequential([
        Conv2D(16, (3,3), strides=(1,1), activation='relu', input_shape=(256,256,3)),
        BatchNormalization(), MaxPooling2D(),
        Conv2D(32, (3,3), activation='relu'), BatchNormalization(), MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu'), BatchNormalization(), MaxPooling2D(),
        Flatten(), Dense(512, activation='relu'), Dropout(0.09), Dense(1, activation='sigmoid'),
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m.load_weights(weights_path)
    return m

def img_run(pil_img, model_name, eff, eff_art, cnn_w):
    img = pil_img.convert("RGB")
    if model_name == "CNN":
        arr = np.array(img.resize((256, 256)), dtype=np.float32) / 255.0
        return build_cnn(cnn_w).predict(arr.reshape(1, 256, 256, 3), verbose=0)
    elif model_name == "EfficientNet":
        arr = np.expand_dims(np.array(img.resize((300, 300)), dtype=np.float32), 0)
        with _strategy().scope(): return eff.predict(arr, verbose=0)
    elif model_name == "EfficientNet Art":
        arr = np.expand_dims(np.array(img.resize((224, 224)), dtype=np.float32), 0)
        with _strategy().scope(): return eff_art.predict(arr, verbose=0)

def img_interpret(preds):
    s = float(preds[0][0])
    if s > 0.5: return "Real Photo",    s, int(s * 100),       "real"
    else:       return "AI Generated",  s, int((1 - s) * 100), "ai"

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🖼️ AI Image Detector</div>', unsafe_allow_html=True)
st.caption("CNN · EfficientNetB3 · EfficientNet Art — each with its original input resolution.")
st.divider()

if "img_prev" not in st.session_state: st.session_state.img_prev = None
if "img_mk"   not in st.session_state: st.session_state.img_mk   = "imk0"

il, ir = st.columns([1, 1], gap="large")
with il:
    uimg = st.file_uploader("Upload image", type=["png","jpg","jpeg","webp"], key="iup")
    if uimg != st.session_state.img_prev:
        st.session_state.img_mk   = "imk1" if st.session_state.img_mk == "imk0" else "imk0"
        st.session_state.img_prev = uimg
    if uimg:
        ib = uimg.read()
        if len(ib) / (1024**2) > 20: st.error("Max 20 MB."); st.stop()
        st.image(ib, use_container_width=True)
    mn = st.selectbox("Detection model", ["CNN","EfficientNet","EfficientNet Art"],
                      index=None, placeholder="Choose a model…", key=st.session_state.img_mk)
with ir:
    rph = st.empty()
    if uimg and mn:
        try:
            eff, eff_art, cw = load_image_models(img_effnet_path, img_effnet_art_path, img_cnn_weights)
        except FileNotFoundError as e:
            st.error(str(e)); st.stop()
        pil = PILImage.open(io.BytesIO(ib))
        with st.spinner(f"Running {mn}…"):
            preds = img_run(pil, mn, eff, eff_art, cw)
        lbl, sc, cp, cls = img_interpret(preds)
        rph.markdown(result_card_html(lbl, cls, cp, sc, f"<b>Model</b> · {mn}"), unsafe_allow_html=True)

st.markdown('<div class="info-box"><b>CNN</b> 256×256 /255 · <b>EfficientNetB3</b> 300×300 · <b>EfficientNet Art</b> 224×224<br>Score &lt;0.5 → AI Generated · Score ≥0.5 → Real Photo</div>', unsafe_allow_html=True)
