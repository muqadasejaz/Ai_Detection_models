# # ─────────────────────────────────────────────────────────────────────────────
# #  pages/4_🎙️_Audio_Detection.py
# #  Deepfake Audio Detector — Mel spectrogram CNN
# #  Only keras/librosa/matplotlib loads when this page is visited.
# # ─────────────────────────────────────────────────────────────────────────────
# import os
# import tempfile
# import numpy as np
# import streamlit as st

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# st.set_page_config(page_title="Audio Detection · DeepSentinel", page_icon="🎙️", layout="wide")

# from utils import inject_css, result_card_html
# inject_css()

# # ── Sidebar ───────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("## 🛡️ DeepSentinel")
#     st.caption("AI & Deepfake Detection Suite")
#     st.divider()
#     st.markdown("### 📁 Model Path")
#     audio_model_path = st.text_input("Audio model (.h5)", value=os.environ.get("AUDIO_MODEL_PATH", "deepfake_audio_detector.h5"))

# # ── Constants ─────────────────────────────────────────────────────────────────
# AUD_MELS   = 128
# AUD_FRAMES = 87
# AUD_MAX_MB = 50
# AUD_MAX_S  = 60.0

# # ── Model loaders & helpers ───────────────────────────────────────────────────
# @st.cache_resource(show_spinner="Loading audio model…")
# def load_audio_model(path):
#     import keras
#     if not os.path.exists(path): raise FileNotFoundError(f"Audio model not found: {path}")
#     return keras.models.load_model(path)

# def audio_features(fp):
#     import librosa
#     audio, sr = librosa.load(fp, sr=None, mono=True)
#     dur  = librosa.get_duration(y=audio, sr=sr)
#     mel  = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=AUD_MELS)
#     mdb  = librosa.power_to_db(mel, ref=np.max)
#     mdb_f = (np.pad(mdb, ((0,0),(0,AUD_FRAMES-mdb.shape[1])), mode="reflect")
#              if mdb.shape[1] < AUD_FRAMES else mdb[:, :AUD_FRAMES])
#     return mdb_f[np.newaxis,...], mdb_f, sr, dur

# def plot_wave(fp, sr):
#     import librosa
#     import matplotlib.pyplot as plt
#     audio, _ = librosa.load(fp, sr=sr, mono=True)
#     fig, ax  = plt.subplots(figsize=(8, 2))
#     fig.patch.set_alpha(0); ax.set_facecolor("none")
#     ax.plot(np.linspace(0, len(audio)/sr, len(audio)), audio, color="#a0a8ff", lw=0.6, alpha=0.9)
#     ax.set_xlabel("Time (s)", fontsize=9); ax.set_ylabel("Amplitude", fontsize=9)
#     ax.tick_params(labelsize=8); ax.spines[["top","right"]].set_visible(False)
#     fig.tight_layout(); return fig

# def plot_mel(mdb, sr):
#     import librosa.display
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots(figsize=(8, 3))
#     fig.patch.set_alpha(0); ax.set_facecolor("none")
#     img = librosa.display.specshow(mdb, sr=sr, x_axis="frames", y_axis="mel", ax=ax, cmap="magma")
#     fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
#     ax.set_title("Mel Spectrogram (model input)", fontsize=10)
#     ax.tick_params(labelsize=8); fig.tight_layout(); return fig

# # ── UI ────────────────────────────────────────────────────────────────────────
# st.markdown('<div class="hero-title">🎙️  Audio Detector</div>', unsafe_allow_html=True)
# st.caption("Mel spectrogram CNN · WAV, MP3, FLAC, OGG · reflect padding · waveform + spectrogram visualization.")
# st.divider()

# sw   = st.toggle("Show waveform",        value=True,  key="asw")
# sm   = st.toggle("Show mel spectrogram", value=True,  key="asm")
# uaud = st.file_uploader("Upload audio",  type=["wav","mp3","flac","ogg"], key="aup")

# if uaud:
#     ab  = uaud.read()
#     amb = len(ab) / (1024**2)
#     if amb > AUD_MAX_MB: st.error(f"File too large ({amb:.1f} MB). Max {AUD_MAX_MB} MB."); st.stop()
#     st.audio(uaud)
#     suf = "." + uaud.name.rsplit(".", 1)[-1]
#     with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as tmp:
#         tmp.write(ab); tp = tmp.name
#     try:
#         try:    amod = load_audio_model(audio_model_path)
#         except FileNotFoundError as e: st.error(str(e)); st.stop()
#         with st.spinner("Extracting features…"):
#             feats, mdb, sr, dur = audio_features(tp)
#         if dur > AUD_MAX_S:
#             st.warning(f"Audio is {dur:.1f}s — only first ~{AUD_FRAMES*512/sr:.1f}s used for prediction.")
#         c1, c2, c3 = st.columns(3)
#         c1.metric("Duration",    f"{dur:.2f}s")
#         c2.metric("Sample rate", f"{sr:,} Hz")
#         c3.metric("File size",   f"{amb:.2f} MB")
#         with st.spinner("Running model…"):
#             pred = float(amod.predict(feats, verbose=0)[0][0])
#         fake_a = pred >= 0.5
#         conf_a = pred if fake_a else 1 - pred
#         lbl_a  = "FAKE" if fake_a else "REAL"
#         cls_a  = "fake" if fake_a else "real"
#         st.divider()
#         rc, bc = st.columns([1, 2])
#         with rc:
#             st.markdown(result_card_html(lbl_a, cls_a, int(conf_a*100), pred, f"<b>File</b> · {uaud.name}"), unsafe_allow_html=True)
#         with bc:
#             st.write("**Score breakdown**")
#             st.progress(float(pred),   text=f"Fake: {pred:.1%}")
#             st.progress(float(1-pred), text=f"Real: {1-pred:.1%}")
#             certainty = ("Very high" if conf_a > 0.90 else "High" if conf_a > 0.75
#                          else "Moderate" if conf_a > 0.60 else "Low — borderline")
#             st.caption(f"Certainty: {certainty}")
#         if sw:
#             st.divider(); st.write("**Waveform**")
#             st.pyplot(plot_wave(tp, sr), use_container_width=True)
#         if sm:
#             st.write("**Mel Spectrogram**")
#             st.pyplot(plot_mel(mdb, sr), use_container_width=True)
#             st.caption("AI audio often shows unusually smooth or repetitive frequency patterns.")
#     except Exception as e:
#         st.error(f"Analysis failed: {e}"); st.exception(e)
#     finally:
#         if os.path.exists(tp): os.unlink(tp)

# st.markdown(f'<div class="info-box"><b>Input shape:</b> (1, {AUD_MELS}, {AUD_FRAMES}) mel dB · <b>Padding:</b> reflect<br><b>Formats:</b> WAV · MP3 · FLAC · OGG · Max {AUD_MAX_MB} MB · {AUD_MAX_S:.0f}s full analysis</div>', unsafe_allow_html=True)

# # ─────────────────────────────────────────────────────────────────────────────
# #  pages/4_🎙️_Audio_Detection.py
# #  Deepfake Audio Detector — Mel spectrogram CNN
# #  Only keras/librosa/matplotlib loads when this page is visited.
# # ─────────────────────────────────────────────────────────────────────────────
# import os
# import tempfile

# # FIX: TF_USE_LEGACY_KERAS has been removed.
# # Setting it here is too late — TensorFlow has already been imported by the
# # time this page loads (other pages in the same process imported it first).
# # Env vars that control TF backend selection must be set before ANY TF import.
# # Instead, we use `from tensorflow import keras` directly in load_audio_model()
# # below, which always gives us the TF-bundled Keras 2 regardless of what
# # standalone Keras 3 is doing. This is consistent and version-safe.
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import numpy as np
# import streamlit as st

# st.set_page_config(page_title="Audio Detection · DeepSentinel", page_icon="🎙️", layout="wide")

# from utils import inject_css, result_card_html, ensure_file
# inject_css()

# # ── Sidebar ───────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("## 🛡️ DeepSentinel")
#     st.caption("AI & Deepfake Detection Suite")
#     st.divider()
#     st.markdown("### 📁 Model Path")
#     audio_model_path = st.text_input("Audio model (.h5)", value=os.environ.get("AUDIO_MODEL_PATH", "deepfake_audio_detector.h5"))

# # ── Constants ─────────────────────────────────────────────────────────────────
# AUD_MELS   = 128
# AUD_FRAMES = 87
# AUD_MAX_MB = 50
# AUD_MAX_S  = 60.0

# # ── Model loaders & helpers ───────────────────────────────────────────────────
# @st.cache_resource(show_spinner="Loading audio model…")
# def load_audio_model(path):
#     # FIX: use `from tensorflow import keras` instead of standalone `import keras`.
#     # This gives us the Keras 2 implementation bundled with TensorFlow, which is
#     # what saved this model. It works regardless of page load order and does not
#     # depend on TF_USE_LEGACY_KERAS being set before import time.
#     from tensorflow import keras
#     return keras.models.load_model(ensure_file(path))

# def audio_features(fp):
#     import librosa
#     audio, sr = librosa.load(fp, sr=None, mono=True)
#     dur  = librosa.get_duration(y=audio, sr=sr)
#     mel  = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=AUD_MELS)
#     mdb  = librosa.power_to_db(mel, ref=np.max)
#     if mdb.shape[1] < AUD_FRAMES:
#         mdb_f = np.pad(
#             mdb, ((0, 0), (0, AUD_FRAMES - mdb.shape[1])),
#             mode="constant", constant_values=float(mdb.min()),
#         )
#     else:
#         mdb_f = mdb[:, :AUD_FRAMES]
#     return mdb_f[np.newaxis, ...], mdb_f, sr, dur

# def plot_wave(fp, sr):
#     import librosa
#     import matplotlib.pyplot as plt
#     audio, _ = librosa.load(fp, sr=sr, mono=True)
#     fig, ax  = plt.subplots(figsize=(8, 2))
#     fig.patch.set_alpha(0); ax.set_facecolor("none")
#     ax.plot(np.linspace(0, len(audio)/sr, len(audio)), audio, color="#a0a8ff", lw=0.6, alpha=0.9)
#     ax.set_xlabel("Time (s)", fontsize=9); ax.set_ylabel("Amplitude", fontsize=9)
#     ax.tick_params(labelsize=8); ax.spines[["top","right"]].set_visible(False)
#     fig.tight_layout(); return fig

# def plot_mel(mdb, sr):
#     import librosa.display
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots(figsize=(8, 3))
#     fig.patch.set_alpha(0); ax.set_facecolor("none")
#     img = librosa.display.specshow(mdb, sr=sr, x_axis="frames", y_axis="mel", ax=ax, cmap="magma")
#     fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
#     ax.set_title("Mel Spectrogram (model input)", fontsize=10)
#     ax.tick_params(labelsize=8); fig.tight_layout(); return fig

# # ── UI ────────────────────────────────────────────────────────────────────────
# st.markdown('<div class="hero-title">🎙️  Audio Detector</div>', unsafe_allow_html=True)
# st.caption("Mel spectrogram CNN · WAV, MP3, FLAC, OGG · reflect padding · waveform + spectrogram visualization.")
# st.divider()

# sw   = st.toggle("Show waveform",        value=True,  key="asw")
# sm   = st.toggle("Show mel spectrogram", value=True,  key="asm")
# uaud = st.file_uploader("Upload audio",  type=["wav","mp3","flac","ogg"], key="aup")

# if uaud:
#     ab  = uaud.read()
#     amb = len(ab) / (1024**2)
#     if amb > AUD_MAX_MB: st.error(f"File too large ({amb:.1f} MB). Max {AUD_MAX_MB} MB."); st.stop()
#     st.audio(uaud)
#     suf = "." + uaud.name.rsplit(".", 1)[-1]
#     with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as tmp:
#         tmp.write(ab); tp = tmp.name
#     try:
#         try:    amod = load_audio_model(audio_model_path)
#         except FileNotFoundError as e: st.error(str(e)); st.stop()
#         with st.spinner("Extracting features…"):
#             feats, mdb, sr, dur = audio_features(tp)
#         if dur > AUD_MAX_S:
#             st.warning(f"Audio is {dur:.1f}s — only first ~{AUD_FRAMES*512/sr:.1f}s used for prediction.")
#         c1, c2, c3 = st.columns(3)
#         c1.metric("Duration",    f"{dur:.2f}s")
#         c2.metric("Sample rate", f"{sr:,} Hz")
#         c3.metric("File size",   f"{amb:.2f} MB")
#         with st.spinner("Running model…"):
#             pred = float(amod.predict(feats, verbose=0)[0][0])
#         fake_a = pred >= 0.5
#         conf_a = pred if fake_a else 1 - pred
#         lbl_a  = "FAKE" if fake_a else "REAL"
#         cls_a  = "fake" if fake_a else "real"
#         st.divider()
#         rc, bc = st.columns([1, 2])
#         with rc:
#             st.markdown(result_card_html(lbl_a, cls_a, int(conf_a*100), pred, f"<b>File</b> · {uaud.name}"), unsafe_allow_html=True)
#         with bc:
#             st.write("**Score breakdown**")
#             st.progress(float(pred),   text=f"Fake: {pred:.1%}")
#             st.progress(float(1-pred), text=f"Real: {1-pred:.1%}")
#             certainty = ("Very high" if conf_a > 0.90 else "High" if conf_a > 0.75
#                          else "Moderate" if conf_a > 0.60 else "Low — borderline")
#             st.caption(f"Certainty: {certainty}")
#         if sw:
#             st.divider(); st.write("**Waveform**")
#             st.pyplot(plot_wave(tp, sr), use_container_width=True)
#         if sm:
#             st.write("**Mel Spectrogram**")
#             st.pyplot(plot_mel(mdb, sr), use_container_width=True)
#             st.caption("AI audio often shows unusually smooth or repetitive frequency patterns.")




#  pages/4_🎙️_Audio_Detection.py
#  Deepfake Audio Detector — Mel spectrogram CNN (Sliding Window Version)
# ─────────────────────────────────────────────────────────────────────────────
import os
import tempfile
import numpy as np
import streamlit as st

# Env vars controlling TF backend selection must be set before ANY TF import.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

st.set_page_config(page_title="Audio Detection · DeepSentinel", page_icon="🎙️", layout="wide")

from utils import inject_css, result_card_html, ensure_file
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
AUD_FRAMES = 87       # Fixed frame count expected by your current CNN model
AUD_MAX_MB = 50       # File size ceiling
AUD_MIN_S  = 10.0     # Enforced minimum audio length
AUD_MAX_S  = 60.0     # Enforced maximum audio length

# ── Model loaders & helpers ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading audio model…")
def load_audio_model(path):
    from tensorflow import keras
    return keras.models.load_model(ensure_file(path))

def process_sliding_windows(fp, amod):
    """
    Slices long audio into overlapping segments matching the CNN's exact 
    input footprint (87 frames), evaluates each segment, and averages the results.
    """
    import librosa
    
    # Load original audio sample rate and timeline
    audio, sr = librosa.load(fp, sr=None, mono=True)
    dur = librosa.get_duration(y=audio, sr=sr)
    
    # Check boundaries before heavy calculations
    if dur < AUD_MIN_S or dur > AUD_MAX_S:
        return None, None, sr, dur

    hop_length = 512
    # Calculate how many audio samples correspond to exactly AUD_FRAMES (87)
    samples_per_window = (AUD_FRAMES - 1) * hop_length
    # Use 50% overlap between windows for granular analysis coverage
    step_size = samples_per_window // 2 
    
    predictions = []
    all_mel_segments = []

    # Slide across the time timeline
    for start in range(0, len(audio) - samples_per_window + 1, step_size):
        chunk = audio[start : start + samples_per_window]
        
        # Build mel spectrogram for this specific window
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=AUD_MELS, hop_length=hop_length)
        mdb = librosa.power_to_db(mel, ref=np.max)
        
        # Verify shape integrity before feeding to TensorFlow
        if mdb.shape[1] == AUD_FRAMES:
            all_mel_segments.append(mdb)
            # Reshape matrix to match CNN expectations (1, 128, 87, 1)
            feats = mdb[np.newaxis, ..., np.newaxis]
            pred = float(amod.predict(feats, verbose=0)[0][0])
            predictions.append(pred)

    # Fallback to standard extraction if file length doesn't fit standard step calculation
    if not predictions:
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=AUD_MELS, hop_length=hop_length)
        mdb = librosa.power_to_db(mel, ref=np.max)
        if mdb.shape[1] < AUD_FRAMES:
            mdb_f = np.pad(mdb, ((0, 0), (0, AUD_FRAMES - mdb.shape[1])), mode="constant", constant_values=float(mdb.min()))
        else:
            mdb_f = mdb[:, :AUD_FRAMES]
        all_mel_segments.append(mdb_f)
        feats = mdb_f[np.newaxis, ..., np.newaxis]
        predictions.append(float(amod.predict(feats, verbose=0)[0][0]))

    final_pred = np.mean(predictions)
    # Return the first segment for UI visualization purposes
    return final_pred, all_mel_segments[0], sr, dur

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
    ax.set_title("Mel Spectrogram (Sample window analysis footprint)", fontsize=10)
    ax.tick_params(labelsize=8); fig.tight_layout(); return fig

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎙️  Audio Detector</div>', unsafe_allow_html=True)
st.caption("Deepfake Audio Detection · Supported range: 10 to 60 seconds · Multi-window analysis timeline scan.")
st.divider()

sw   = st.toggle("Show waveform",        value=True,  key="asw")
sm   = st.toggle("Show mel spectrogram", value=True,  key="asm")
uaud = st.file_uploader("Upload audio",  type=["wav","mp3","flac","ogg"], key="aup")

if uaud:
    ab  = uaud.read()
    amb = len(ab) / (1024**2)
    if amb > AUD_MAX_MB: 
        st.error(f"File too large ({amb:.1f} MB). Max allowed size is {AUD_MAX_MB} MB.")
        st.stop()
        
    st.audio(uaud)
    suf = "." + uaud.name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as tmp:
        tmp.write(ab); tp = tmp.name
        
    try:
        try:    
            amod = load_audio_model(audio_model_path)
        except FileNotFoundError as e: 
            st.error(str(e)); st.stop()
            
        with st.spinner("Analyzing timeline composition across entire audio track…"):
            pred, mdb, sr, dur = process_sliding_windows(tp, amod)
            
        # Hard limits check
        if dur < AUD_MIN_S:
            st.error(f"Audio file is too short ({dur:.1f}s). Please upload audio between {AUD_MIN_S}s and {AUD_MAX_S}s.")
            st.stop()
        if dur > AUD_MAX_S:
            st.error(f"Audio file is too long ({dur:.1f}s). Maximum allowed track length is {AUD_MAX_S} seconds.")
            st.stop()
            
        st.info(f"Timeline Check: Evaluated entire {dur:.1f}s file by splitting it into sequential rolling analysis segments.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Duration",    f"{dur:.2f}s")
        c2.metric("Sample rate", f"{sr:,} Hz")
        c3.metric("File size",   f"{amb:.2f} MB")
        
        fake_a = pred >= 0.5
        conf_a = pred if fake_a else 1 - pred
        lbl_a  = "FAKE" if fake_a else "REAL"
        cls_a  = "fake" if fake_a else "real"
        
        st.divider()
        rc, bc = st.columns([1, 2])
        with rc:
            st.markdown(result_card_html(lbl_a, cls_a, int(conf_a*100), pred, f"<b>File</b> · {uaud.name}"), unsafe_allow_html=True)
        with bc:
            st.write("**Integrated Timeline Breakdown Score**")
            st.progress(float(pred),   text=f"Fake Probability: {pred:.1%}")
            st.progress(float(1-pred), text=f"Real Probability: {1-pred:.1%}")
            certainty = ("Very high" if conf_a > 0.90 else "High" if conf_a > 0.75
                         else "Moderate" if conf_a > 0.60 else "Low — borderline")
            st.caption(f"Confidence Rating: {certainty}")
            
        if sw:
            st.divider(); st.write("**Waveform**")
            st.pyplot(plot_wave(tp, sr), use_container_width=True)
        if sm:
            st.write("**Mel Spectrogram**")
            st.pyplot(plot_mel(mdb, sr), use_container_width=True)
            st.caption("AI voice generations often leave distinctly repetitive signatures across frequency bands.")
            
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.exception(e)
    finally:
        if os.path.exists(tp): 
            os.unlink(tp)
#     except Exception as e:
#         st.error(f"Analysis failed: {e}"); st.exception(e)
#     finally:
#         if os.path.exists(tp): os.unlink(tp)
