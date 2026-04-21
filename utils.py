# ─────────────────────────────────────────────────────────────────────────────
#  utils.py — Shared helpers for DeepSentinel
#  Imported by app.py and every pages/*.py file
# ─────────────────────────────────────────────────────────────────────────────
import os
import streamlit as st

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
DRIVE_FILES = {
    "lstm_main.h5":                             "lstm_main",
    "tokenizer.pkl":                            "tokenizer",
    "lstm_kfold.h5":                            "lstm_kfold",
    "tokenizer_lstm_best_kfold.pkl":            "tokenizer_kfold",
    "efficientnetb3_binary_classifier_8.h5":    "effnet",
    "EfficientNet_fine_tune_art_model.h5":      "effnet_art",
    "model_weights.weights.h5":                 "cnn_weights",
    "deepfake_audio_detector.h5":               "audio_model",
    "model_97_acc_100_frames_FF_data.pt":       "vid_97_100",
    "model_97_acc_80_frames_FF_data.pt":        "vid_97_80",
    "model_97_acc_60_frames_FF_data.pt":        "vid_97_60",
    "model_95_acc_40_frames_FF_data.pt":        "vid_95_40",
    "model_93_acc_100_frames_celeb_FF_data.pt": "vid_93_100",
}

# Hardcoded IDs for lstm models
HARDCODED_IDS = {
    "lstm_main.h5":  "1jhO_HlI8CEL0VgXxih51SqsE9h60pxir",
    "lstm_kfold.h5": "1-1e5psNK8Nb5wDUyg8vgpdM0Zi-8W6Mp",
}

@st.cache_resource(show_spinner=False)
def download_models():
    try:
        import gdown
    except ImportError:
        st.error("gdown not installed — add `gdown` to requirements.txt")
        st.stop()

    gdrive_secrets = st.secrets.get("gdrive", {})

    missing = []
    for fname, key in DRIVE_FILES.items():
        if os.path.exists(fname):
            continue
        if fname in HARDCODED_IDS:
            missing.append((fname, HARDCODED_IDS[fname]))
        elif key in gdrive_secrets:
            missing.append((fname, gdrive_secrets[key]))

    if not missing:
        return

    bar = st.progress(0, text="⬇️  Downloading model weights from Google Drive…")
    for i, (fname, file_id) in enumerate(missing):
        bar.progress(i / len(missing), text=f"⬇️  Downloading {fname}…")
        gdown.download(id=file_id, output=fname, quiet=False)
    bar.progress(1.0, text="✅  All models ready!")
    bar.empty()
