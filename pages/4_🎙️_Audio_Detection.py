# ─────────────────────────────────────────────────────────────────────────────
#  pages/4_🎙️_Audio_Detection.py
#  Deepfake Audio Detector — Mel spectrogram CNN
#  Only keras/librosa/matplotlib loads when this page is visited.
# ─────────────────────────────────────────────────────────────────────────────
import os
import tempfile
import numpy as np
import streamlit as st

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

st.set_page_config(page_title="Audio Detection · DeepSentinel", page_icon="🎙️", layout="wide")

from utils import inject_css, result_card_html
inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ DeepSentinel")
    st.caption("AI & Deepfake Detection Suite")
    st.divider()
    st.markdown("### 📁 Model Path")
    audio_model_path = st.text_input("Audio model (.h5)", value=os.environ.get("AUDIO_MODEL_PATH", "deepfake_audio_detector.h5"))

# ── Constants ─────────────────────────────────────────────────────────────────
AUD_MELS   = 128
AUD_FRAMES = 87
AUD_MAX_MB = 50
AUD_MAX_S  = 60.0

# ── Model loaders & helpers ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading audio model…")
def load_audio_model(path):
    import keras
    if not os.path.exists(path): raise FileNotFoundError(f"Audio model not found: {path}")
    return keras.models.load_model(path)

def audio_features(fp):
    import librosa
    audio, sr = librosa.load(fp, sr=None, mono=True)
    dur  = librosa.get_duration(y=audio, sr=sr)
    mel  = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=AUD_MELS)
    mdb  = librosa.power_to_db(mel, ref=np.max)
    mdb_f = (np.pad(mdb, ((0,0),(0,AUD_FRAMES-mdb.shape[1])), mode="reflect")
             if mdb.shape[1] < AUD_FRAMES else mdb[:, :AUD_FRAMES])
    return mdb_f[np.newaxis,...], mdb_f, sr, dur

def plot_wave(fp, sr):
    import librosa
    import matplotlib.pyplot as plt
    audio, _ = librosa.load(fp, sr=sr, mono=True)
    fig, ax  = plt.subplots(figsize=(8, 2))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.plot(np.linspace(0, len(audio)/sr, len(audio)), audio, color="#a0a8ff", lw=0.6, alpha=0.9)
    ax.set_xlabel("Time (s)", fontsize=9); ax.set_ylabel("Amplitude", fontsize=9)
    ax.tick_params(labelsize=8); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); return fig

def plot_mel(mdb, sr):
    import librosa.display
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    img = librosa.display.specshow(mdb, sr=sr, x_axis="frames", y_axis="mel", ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    ax.set_title("Mel Spectrogram (model input)", fontsize=10)
    ax.tick_params(labelsize=8); fig.tight_layout(); return fig

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎙️  Audio Detector</div>', unsafe_allow_html=True)
st.caption("Mel spectrogram CNN · WAV, MP3, FLAC, OGG · reflect padding · waveform + spectrogram visualization.")
st.divider()

sw   = st.toggle("Show waveform",        value=True,  key="asw")
sm   = st.toggle("Show mel spectrogram", value=True,  key="asm")
uaud = st.file_uploader("Upload audio",  type=["wav","mp3","flac","ogg"], key="aup")

if uaud:
    ab  = uaud.read()
    amb = len(ab) / (1024**2)
    if amb > AUD_MAX_MB: st.error(f"File too large ({amb:.1f} MB). Max {AUD_MAX_MB} MB."); st.stop()
    st.audio(uaud)
    suf = "." + uaud.name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as tmp:
        tmp.write(ab); tp = tmp.name
    try:
        try:    amod = load_audio_model(audio_model_path)
        except FileNotFoundError as e: st.error(str(e)); st.stop()
        with st.spinner("Extracting features…"):
            feats, mdb, sr, dur = audio_features(tp)
        if dur > AUD_MAX_S:
            st.warning(f"Audio is {dur:.1f}s — only first ~{AUD_FRAMES*512/sr:.1f}s used for prediction.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Duration",    f"{dur:.2f}s")
        c2.metric("Sample rate", f"{sr:,} Hz")
        c3.metric("File size",   f"{amb:.2f} MB")
        with st.spinner("Running model…"):
            pred = float(amod.predict(feats, verbose=0)[0][0])
        fake_a = pred >= 0.5
        conf_a = pred if fake_a else 1 - pred
        lbl_a  = "FAKE" if fake_a else "REAL"
        cls_a  = "fake" if fake_a else "real"
        st.divider()
        rc, bc = st.columns([1, 2])
        with rc:
            st.markdown(result_card_html(lbl_a, cls_a, int(conf_a*100), pred, f"<b>File</b> · {uaud.name}"), unsafe_allow_html=True)
        with bc:
            st.write("**Score breakdown**")
            st.progress(float(pred),   text=f"Fake: {pred:.1%}")
            st.progress(float(1-pred), text=f"Real: {1-pred:.1%}")
            certainty = ("Very high" if conf_a > 0.90 else "High" if conf_a > 0.75
                         else "Moderate" if conf_a > 0.60 else "Low — borderline")
            st.caption(f"Certainty: {certainty}")
        if sw:
            st.divider(); st.write("**Waveform**")
            st.pyplot(plot_wave(tp, sr), use_container_width=True)
        if sm:
            st.write("**Mel Spectrogram**")
            st.pyplot(plot_mel(mdb, sr), use_container_width=True)
            st.caption("AI audio often shows unusually smooth or repetitive frequency patterns.")
    except Exception as e:
        st.error(f"Analysis failed: {e}"); st.exception(e)
    finally:
        if os.path.exists(tp): os.unlink(tp)

st.markdown(f'<div class="info-box"><b>Input shape:</b> (1, {AUD_MELS}, {AUD_FRAMES}) mel dB · <b>Padding:</b> reflect<br><b>Formats:</b> WAV · MP3 · FLAC · OGG · Max {AUD_MAX_MB} MB · {AUD_MAX_S:.0f}s full analysis</div>', unsafe_allow_html=True)
