# ─────────────────────────────────────────────────────────────────────────────
#  utils.py — Shared helpers for DeepSentinel
#  Imported by app.py and every pages/*.py file
#
#  KEY CHANGE: models are no longer all downloaded up-front. Use ensure_file()
#  inside each page's cached loader so each weight downloads ONCE, on demand.
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import pickle

# NOTE: we deliberately do NOT set TF_USE_LEGACY_KERAS. The models in this
# project were saved with mixed Keras versions (the image CNN is Keras 3 — its
# InputLayer uses `batch_shape`, which Keras 2 cannot read — while the text
# tokenizer/LSTMs are Keras 2). load_keras_model() below tries both backends so
# each model loads with whichever one saved it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import streamlit as st

# ── Global CSS ────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: #0e0f1a !important; border-right: 1px solid #1c1d30; }
[data-testid="stSidebar"] * { color: #9899b8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #d0d1e8 !important; }
[data-testid="stSidebar"] input {
    background: #161728 !important; border: 1px solid #2a2b44 !important;
    color: #d0d1e8 !important; border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.78rem !important;
}
[data-testid="stMetricValue"] { font-family: 'DM Mono', monospace !important; font-size: 1.3rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.72rem !important; color: #5a5b7a !important; letter-spacing: 0.08em; }
.result-card {
    border-radius: 14px; padding: 1.6rem 1.8rem 1.4rem;
    border: 1px solid #1c1d30; background: #0e0f1a;
    position: relative; overflow: hidden; margin-bottom: 1rem;
}
.result-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
.result-card.fake::before  { background: linear-gradient(90deg,#ff3a5c,#ff8c00); }
.result-card.real::before  { background: linear-gradient(90deg,#00d4aa,#0090ff); }
.result-card.ai::before    { background: linear-gradient(90deg,#ff3a5c,#ff8c00); }
.result-card.human::before { background: linear-gradient(90deg,#00d4aa,#0090ff); }
.verdict-eyebrow { font-family:'DM Mono',monospace; font-size:0.62rem; color:#3a3b5a; letter-spacing:0.18em; text-transform:uppercase; margin-bottom:0.35rem; }
.verdict-text { font-family:'DM Sans',sans-serif; font-weight:700; font-size:2.2rem; letter-spacing:-0.5px; line-height:1; margin-bottom:0.7rem; }
.verdict-text.fake, .verdict-text.ai    { color:#ff4060; }
.verdict-text.real, .verdict-text.human { color:#00d4aa; }
.conf-row  { display:flex; align-items:center; gap:0.8rem; margin-bottom:0.4rem; }
.conf-bg   { flex:1; height:5px; background:#181928; border-radius:3px; overflow:hidden; }
.conf-fill { height:100%; border-radius:3px; }
.conf-fill.fake, .conf-fill.ai    { background:linear-gradient(90deg,#ff3a5c,#ff8c00); }
.conf-fill.real, .conf-fill.human { background:linear-gradient(90deg,#00d4aa,#0090ff); }
.conf-pct  { font-family:'DM Mono',monospace; font-size:0.88rem; font-weight:500; color:#d0d1e8; min-width:3rem; text-align:right; }
.meta-line { font-family:'DM Mono',monospace; font-size:0.65rem; color:#2e2f4a; line-height:1.9; margin-top:0.5rem; }
.meta-line b { color:#44456a; }
.frame-label { font-family:'DM Mono',monospace; font-size:0.58rem; padding:0.25rem 0.4rem; text-align:center; border-radius:0 0 8px 8px; }
.frame-label.fake { color:#ff4060; background:rgba(255,60,80,0.08); }
.frame-label.real { color:#00d4aa; background:rgba(0,212,170,0.08); }
.tl-bar  { display:flex; height:7px; border-radius:4px; overflow:hidden; gap:1px; margin:0.6rem 0; }
.tl-seg  { flex:1; }
.tl-fake { background:#ff3a5c; }
.tl-real { background:#00d4aa; }
.model-cards { display:flex; gap:0.8rem; flex-wrap:wrap; margin:1rem 0; }
.model-card  { background:#0c0d1e; border:1px solid #1c1d30; border-radius:10px; padding:0.7rem 1rem; flex:1; min-width:150px; }
.mc-name  { font-family:'DM Mono',monospace; font-size:0.62rem; color:#3a3b5a; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.25rem; }
.mc-score { font-family:'DM Sans',sans-serif; font-weight:700; font-size:1.4rem; letter-spacing:-0.5px; }
.mc-score.fake { color:#ff4060; }
.mc-score.real { color:#00d4aa; }
.mc-conf  { font-family:'DM Mono',monospace; font-size:0.6rem; color:#2e2f4a; }
.section-label-vid { font-family:'DM Mono',monospace; font-size:0.63rem; color:#3a3b5a; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.35rem; }
.info-box { background:#0a0b18; border:1px solid #181928; border-radius:10px; padding:0.9rem 1.1rem; font-family:'DM Mono',monospace; font-size:0.68rem; color:#2e2f4a; line-height:1.8; margin-top:1.5rem; }
.info-box b { color:#44456a; }
.hero-title { font-family:'DM Sans',sans-serif; font-weight:700; font-size:1.5rem; color:#d0d1e8; margin-bottom:0.2rem; }
.hero-sub   { font-family:'DM Mono',monospace; font-size:0.7rem; color:#3a3b5a; letter-spacing:0.08em; }
</style>
"""


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def result_card_html(label, css_cls, conf_pct, raw_score, meta=""):
    return f"""
<div class="result-card {css_cls}">
    <div class="verdict-eyebrow">Detection Result</div>
    <div class="verdict-text {css_cls}">{label}</div>
    <div class="conf-row">
        <div class="conf-bg"><div class="conf-fill {css_cls}" style="width:{conf_pct}%"></div></div>
        <div class="conf-pct">{conf_pct}%</div>
    </div>
    <div class="meta-line"><b>Raw score</b> · {raw_score:.4f}<br>{meta}</div>
</div>"""


# ── Model file map ─────────────────────────────────────────────────────────────
#  fname on disk  ->  key used in [gdrive] secrets
#  FIX: the CNN entry is now "cnn_detection.h5" so it matches the filename used
#  in pages/2 (was "cnn_detection" with no extension -> downloaded twice).
DRIVE_FILES = {
    "lstm_main.h5":                             "lstm_main",
    "tokenizer.pkl":                            "tokenizer",
    "lstm_kfold.h5":                            "lstm_kfold",
    "tokenizer_lstm_best_kfold.pkl":            "tokenizer_kfold",
    "cnn_detection.h5":                         "cnn_detection",
    "deepfake_audio_detector.h5":               "audio_model",
    "model_97_acc_100_frames_FF_data.pt":       "vid_97_100",
    "model_97_acc_80_frames_FF_data.pt":        "vid_97_80",
    "model_97_acc_60_frames_FF_data.pt":        "vid_97_60",
    "model_95_acc_40_frames_FF_data.pt":        "vid_95_40",
    "model_93_acc_100_frames_celeb_FF_data.pt": "vid_93_100",
}

# Hardcoded IDs for the two LSTM weight files (not kept in secrets).
HARDCODED_IDS = {
    "lstm_main.h5":  "1jhO_HlI8CEL0VgXxih51SqsE9h60pxir",
    "lstm_kfold.h5": "1-1e5psNK8Nb5wDUyg8vgpdM0Zi-8W6Mp",
}

# A weight file is considered valid only if it is on disk AND non-trivial in size
# (guards against half-finished / HTML-error downloads).
_MIN_VALID_BYTES = 1_000_000


def _resolve_file_id(fname: str) -> str:
    """Find the Google Drive file id for a given filename."""
    if fname in HARDCODED_IDS:
        return HARDCODED_IDS[fname]
    key = DRIVE_FILES.get(fname)
    if key is None:
        return ""
    gdrive_secrets = st.secrets.get("gdrive", {})
    # also tolerate a flat (non-[gdrive]) layout
    return (gdrive_secrets.get(key) or st.secrets.get(key) or "").strip()


@st.cache_resource(show_spinner=False)
def ensure_file(fname: str) -> str:
    """
    Make sure a single weight file is present on disk, downloading it ONCE on
    first use. Returns the local path.

    Cached with @st.cache_resource so the download runs at most once per file
    for the whole app lifetime — and crucially, nothing is fetched until the
    page that actually needs the file calls this.
    """
    if os.path.exists(fname) and os.path.getsize(fname) > _MIN_VALID_BYTES:
        return fname

    try:
        import gdown
    except ImportError:
        st.error("`gdown` not installed — add `gdown>=5.1.0` to requirements.txt and redeploy.")
        st.stop()

    file_id = _resolve_file_id(fname)
    if not file_id:
        st.error(
            f"No Google Drive id found for `{fname}`. Add it under `[gdrive]` in "
            f"Streamlit secrets (key: `{DRIVE_FILES.get(fname, fname)}`)."
        )
        st.stop()

    with st.spinner(f"⬇️ Downloading {fname} (first use only)…"):
        gdown.download(id=file_id, output=fname, quiet=False)

    if not os.path.exists(fname) or os.path.getsize(fname) < _MIN_VALID_BYTES:
        st.error(
            f"Download of `{fname}` produced an invalid/too-small file. "
            "Check the file id and that it is shared as 'Anyone with the link can view'."
        )
        st.stop()

    return fname


# ── Cross-version Keras loading ────────────────────────────────────────────────
#  Models here were saved with different Keras versions, so we try the Keras 3
#  loader first and fall back to the legacy Keras 2 loader (tf-keras). Both
#  packages are installed; TF_USE_LEGACY_KERAS is intentionally left unset so
#  `import keras` == Keras 3 and `import tf_keras` == Keras 2 are both available.
def load_keras_model(path, compile=False):
    """Load an .h5/.keras model with whichever Keras backend can read it."""
    errors = []
    try:
        import keras                              # Keras 3 (ships with TF 2.20)
        return keras.models.load_model(path, compile=compile)
    except Exception as e:
        errors.append(f"Keras 3 → {type(e).__name__}: {e}")
    try:
        import tf_keras                           # legacy Keras 2
        return tf_keras.models.load_model(path, compile=compile)
    except Exception as e:
        errors.append(f"Keras 2 (tf-keras) → {type(e).__name__}: {e}")
    raise RuntimeError(
        f"Could not load `{path}` with either Keras backend:\n  - "
        + "\n  - ".join(errors)
    )


class _TokenizerCompatUnpickler(pickle.Unpickler):
    """Redirect a pickled Keras Tokenizer to a backend that still defines it.
    Keras 3 removed keras.preprocessing.text.Tokenizer, so old pickles must be
    pointed at the tf-keras (Keras 2) implementation."""
    def find_class(self, module, name):
        if name == "Tokenizer" and "preprocessing" in module:
            for mod in ("tf_keras.preprocessing.text",
                        "keras.preprocessing.text",
                        "keras.src.legacy.preprocessing.text",
                        "keras_preprocessing.text"):
                try:
                    __import__(mod)
                    return getattr(sys.modules[mod], "Tokenizer")
                except Exception:
                    continue
        return super().find_class(module, name)


def load_tokenizer_pickle(path):
    """Unpickle a Keras Tokenizer saved under any Keras version."""
    with open(path, "rb") as f:
        return _TokenizerCompatUnpickler(f).load()


def pad_sequences_compat(sequences, **kwargs):
    """pad_sequences that works under Keras 3 or legacy Keras 2."""
    try:
        from keras.utils import pad_sequences as _ps                              # Keras 3
    except Exception:
        from tensorflow.keras.preprocessing.sequence import pad_sequences as _ps  # Keras 2
    return _ps(sequences, **kwargs)
