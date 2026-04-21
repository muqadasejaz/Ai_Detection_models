# ─────────────────────────────────────────────────────────────────────────────
#  DeepSentinel — Combined AI / Deepfake Detection Suite
#  Tabs: Text · Image · Video · Audio
#
#  DEPLOYMENT NOTE:
#  Model weights are NOT stored in this repo (too large for GitHub).
#  They are downloaded automatically from Google Drive on first run.
#  Add your Drive file IDs in Streamlit Cloud → App Settings → Secrets:
#
#  [gdrive]
#  lstm_main            = "YOUR_FILE_ID"
#  tokenizer            = "YOUR_FILE_ID"
#  lstm_kfold           = "YOUR_FILE_ID"
#  tokenizer_kfold      = "YOUR_FILE_ID"
#  effnet               = "YOUR_FILE_ID"
#  effnet_art           = "YOUR_FILE_ID"
#  cnn_weights          = "YOUR_FILE_ID"
#  audio_model          = "YOUR_FILE_ID"
#  vid_97_100           = "YOUR_FILE_ID"
#  vid_97_80            = "YOUR_FILE_ID"
#  vid_97_60            = "YOUR_FILE_ID"
#  vid_95_40            = "YOUR_FILE_ID"
#  vid_93_100           = "YOUR_FILE_ID"
# ─────────────────────────────────────────────────────────────────────────────
import os
import io
import math
import pickle
import tempfile

import numpy as np
import streamlit as st

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="DeepSentinel · AI Detection Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GOOGLE DRIVE MODEL DOWNLOADER
#  Runs once at startup via @st.cache_resource.
#  File IDs live in st.secrets["gdrive"] — set in Streamlit Cloud UI only.
#  NEVER hardcode file IDs or API keys directly in this file.
# ═══════════════════════════════════════════════════════════════════════════════

# Map: local filename  →  key inside [gdrive] secrets section
_DRIVE_FILES = {
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

@st.cache_resource(show_spinner=False)
def _download_all_models():
    try:
        import gdown
    except ImportError:
        st.error("gdown not installed — add `gdown` to requirements.txt")
        st.stop()

    gdrive_secrets = st.secrets.get("gdrive", {})
    if not gdrive_secrets:
        st.warning(
            "⚠️  No [gdrive] secrets configured. "
            "Go to Streamlit Cloud → your app → Settings → Secrets and add your Drive file IDs."
        )
        return

    # Hardcoded IDs for lstm models
    _HARDCODED = {
        "lstm_main.h5":  "1jhO_HlI8CEL0VgXxih51SqsE9h60pxir",
        "lstm_kfold.h5": "1-1e5psNK8Nb5wDUyg8vgpdM0Zi-8W6Mp",
    }

    missing = []
    for fname, key in _DRIVE_FILES.items():
        if os.path.exists(fname):
            continue
        if fname in _HARDCODED:
            missing.append((fname, _HARDCODED[fname]))
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

_download_all_models()


# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
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
[data-testid="stTabs"] button {
    font-family: 'DM Mono', monospace !important; font-size: 0.78rem !important;
    letter-spacing: 0.06em !important; color: #5a5b7a !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.5rem 1.2rem !important; transition: color 0.2s !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a0a8ff !important; border-bottom-color: #a0a8ff !important;
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
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ DeepSentinel")
    st.caption("AI & Deepfake Detection Suite")
    st.divider()
    st.markdown("### 📁 Model Paths")
    st.caption("Models are auto-downloaded from Google Drive on startup.")

    with st.expander("Text models"):
        text_model_standard  = st.text_input("Standard model (.h5)",      value=os.environ.get("TEXT_MODEL_STANDARD", "lstm_main.h5"))
        text_tok_standard    = st.text_input("Standard tokenizer (.pkl)", value=os.environ.get("TEXT_TOK_STANDARD",   "tokenizer.pkl"))
        text_model_pro       = st.text_input("Pro model (.h5)",           value=os.environ.get("TEXT_MODEL_PRO",      "lstm_kfold.h5"))
        text_tok_pro         = st.text_input("Pro tokenizer (.pkl)",      value=os.environ.get("TEXT_TOK_PRO",        "tokenizer_lstm_best_kfold.pkl"))

    with st.expander("Image models"):
        img_effnet_path     = st.text_input("EfficientNetB3 (.h5)",  value=os.environ.get("IMG_EFFNET_PATH",     "efficientnetb3_binary_classifier_8.h5"))
        img_effnet_art_path = st.text_input("EfficientNet Art (.h5)",value=os.environ.get("IMG_EFFNET_ART_PATH", "EfficientNet_fine_tune_art_model.h5"))
        img_cnn_weights     = st.text_input("CNN weights (.h5)",     value=os.environ.get("IMG_CNN_WEIGHTS",     "model_weights.weights.h5"))

    with st.expander("Video models"):
        st.caption("5 ResNeXt+LSTM models — auto-downloaded from Drive.")
        vid_model_dir = st.text_input(
            "Model directory",
            value=os.environ.get("VID_MODEL_DIR", "."),
            help="Directory where .pt files are saved (default: same folder as app.py)"
        )

    with st.expander("Audio model"):
        audio_model_path = st.text_input("Audio model (.h5)", value=os.environ.get("AUDIO_MODEL_PATH", "deepfake_audio_detector.h5"))

    st.divider()
    st.markdown("### 🔑 API Keys")
    openai_api_key = st.text_input(
        "OpenAI API key (Text Humanizer)",
        type="password", placeholder="sk-...",
        value=os.environ.get("OPENAI_API_KEY", ""),
    )
    if openai_api_key:
        st.success("Humanizer ready ✅")
    st.divider()
    st.caption("Keras · PyTorch · librosa · GPT-4o")


# ── Shared result card helper ─────────────────────────────────────────────────
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


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🛡️ DeepSentinel</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI &amp; Deepfake Detection Suite &nbsp;·&nbsp; Text · Image · Video · Audio</div>', unsafe_allow_html=True)
st.divider()

tab_text, tab_image, tab_video, tab_audio = st.tabs(
    ["📝  Text", "🖼️  Image", "🎬  Video", "🎙️  Audio"]
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — TEXT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_text:
    TEXT_MAX_LEN = 1000
    TEXT_MODELS  = {
        "⚡ Standard": {"model_path": text_model_standard, "tokenizer_path": text_tok_standard,
                        "description": "Single-pass training. Fast and reliable for most texts."},
        "🏆 Pro":      {"model_path": text_model_pro,      "tokenizer_path": text_tok_pro,
                        "description": "5-fold cross-validation. Better on tricky or edge-case texts."},
    }
    H_PROMPTS = [
        """You are a professional human writer. Rewrite the AI-generated text so no detector flags it,
keeping every fact and meaning intact. Vary sentence length aggressively. Use contractions everywhere.
Replace formal words with conversational ones. Delete filler phrases (Furthermore, Moreover, It is important to note,
Delve into, Leverage, etc.). Let paragraphs be uneven. Occasionally start sentences with But/And/So.
OUTPUT: only the rewritten text.""",
        """You are rewriting text an AI detector still flagged. Go further: restructure sentences completely,
split any sentence over 20 words, merge short consecutive sentences, start sentences with verbs/conditions/questions,
add natural imperfections (dashes, parentheses, one-sentence paragraphs). Keep all facts.
OUTPUT: only the rewritten text.""",
        """CRITICAL REWRITE — flagged twice. Treat input as a rough outline and rewrite from scratch in your own voice.
Keep every fact, statistic, name, conclusion. Vary paragraph length heavily. Contractions throughout.
At least 30% of sentences must NOT start with The/This/It/A. FORBIDDEN: Furthermore, Moreover, In addition,
In conclusion, Leverage, Utilize, Delve into, Landscape, Paradigm, It is important to note.
OUTPUT: only the rewritten text.""",
    ]

    @st.cache_resource(show_spinner="Loading text model…")
    def load_text_model(path):
        from tensorflow.keras.models import load_model as _lm
        return _lm(path)

    @st.cache_resource(show_spinner="Loading tokenizer…")
    def load_tokenizer(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def text_predict(text, model, tokenizer):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=TEXT_MAX_LEN, padding="post", truncating="post")
        return float(model.predict(pad, verbose=0)[0][0])

    def text_verdict(prob):
        ai = prob * 100; hu = (1 - prob) * 100
        if prob > 0.75:   v, c = "🤖 Likely AI-Generated",     "#e74c3c"
        elif prob > 0.5:  v, c = "🤖 Possibly AI-Generated",   "#e67e22"
        elif prob > 0.25: v, c = "✍️ Possibly Human-Written",  "#2980b9"
        else:             v, c = "✍️ Likely Human-Written",     "#27ae60"
        return v, c, ai, hu

    def _openai_call(client, prompt, text, temp):
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":prompt},
                      {"role":"user","content":f"Rewrite this text:\n\n{text}"}],
            temperature=temp, top_p=0.95, frequency_penalty=0.5, presence_penalty=0.4,
        )
        return r.choices[0].message.content.strip()

    def humanize(text, api_key, model, tokenizer,
                 threshold=0.45, aggressiveness="Balanced", status_cb=None):
        from openai import OpenAI
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        client = OpenAI(api_key=api_key)
        temps  = [0.88, 0.92, 0.95]
        cfg    = {"Light":{"start":0,"max":2},"Balanced":{"start":0,"max":3},"Aggressive":{"start":1,"max":3}}
        s_idx  = cfg[aggressiveness]["start"]
        mx     = cfg[aggressiveness]["max"]
        def score(t):
            seq = tokenizer.texts_to_sequences([t])
            pad = pad_sequences(seq, maxlen=TEXT_MAX_LEN, padding="post", truncating="post")
            return float(model.predict(pad, verbose=0)[0][0])
        best_t, best_s = text, score(text)
        cur = text; results = []
        for i in range(s_idx, s_idx + mx):
            idx = min(i, len(H_PROMPTS)-1)
            pn  = i - s_idx + 1
            if status_cb: status_cb(f"Pass {pn}: {['light','structural','deep'][idx]} rewrite…")
            rw = _openai_call(client, H_PROMPTS[idx], cur, temps[idx])
            ns = score(rw)
            results.append({"pass":pn,"score":ns,"text":rw})
            if ns < best_s: best_s, best_t = ns, rw
            if ns <= threshold: break
            cur = rw
        return best_t, best_s, results

    for k, v in [("tp",None),("tl",None),("th",None),("ths",None),("tpr",[])]:
        if k not in st.session_state: st.session_state[k] = v

    st.subheader("📝 AI Text Detector & Humanizer")
    st.caption("Detect AI-generated text, then optionally rewrite it with GPT-4o.")
    st.divider()
    sel = st.radio("Detection model", list(TEXT_MODELS.keys()), index=0, horizontal=True)
    mi  = TEXT_MODELS[sel]
    st.caption(f"ℹ️ {mi['description']}")
    st.divider()
    tin = st.text_area("Paste your text here", height=200, placeholder="Type or paste text to analyze…", max_chars=5000)
    wc  = len(tin.split()) if tin.strip() else 0
    st.caption(f"{wc} words · {len(tin)} characters")

    if st.button("🔍 Analyze Text", type="primary", use_container_width=True, key="ta"):
        if not tin.strip(): st.warning("Enter some text first.")
        elif wc < 10:       st.warning("Paste at least 10 words for a reliable result.")
        else:
            st.session_state.th = st.session_state.ths = None
            st.session_state.tl = tin
            with st.spinner("Analyzing…"):
                try:
                    _m = load_text_model(mi["model_path"])
                    _t = load_tokenizer(mi["tokenizer_path"])
                    st.session_state.tp = text_predict(tin, _m, _t)
                except FileNotFoundError as e:
                    st.error(f"File not found: `{e.filename}` — check sidebar paths."); st.stop()

    if st.session_state.tp is not None and st.session_state.tl == tin:
        prob = st.session_state.tp
        vd, vc, ai_pct, hu_pct = text_verdict(prob)
        st.divider()
        st.markdown(f"<h3 style='color:{vc};'>{vd}</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("🤖 AI Probability",    f"{ai_pct:.1f}%")
        c2.metric("✍️ Human Probability", f"{hu_pct:.1f}%")
        st.progress(prob, text=f"AI score: {ai_pct:.1f}%")
        with st.expander("How to read this"):
            st.markdown("- **>75%** Very likely AI · **50–75%** Leans AI · **25–50%** Leans human · **<25%** Very likely human\n\nAnalyzes first ~1,000 words.")

        if prob > 0.5:
            st.divider()
            st.subheader("✍️ Humanize This Text")
            st.caption("Multi-pass GPT-4o rewrite, auto re-scored after each pass.")
            ca, cb = st.columns([2,1])
            with ca:
                agg = st.select_slider("Rewrite strength", ["Light","Balanced","Aggressive"], value="Balanced")
            with cb:
                st.markdown("<br>", unsafe_allow_html=True)
                hbtn = st.button("✨ Humanize", type="primary", use_container_width=True,
                                 disabled=not openai_api_key, key="hbtn")
            if not openai_api_key: st.warning("Add your OpenAI API key in the sidebar.")
            if hbtn and openai_api_key:
                sb = st.empty()
                try:
                    _m = load_text_model(mi["model_path"])
                    _t = load_tokenizer(mi["tokenizer_path"])
                    bt, bs, pr = humanize(tin, openai_api_key, _m, _t,
                                          aggressiveness=agg,
                                          status_cb=lambda m: sb.info(f"⏳ {m}"))
                    sb.empty()
                    st.session_state.th  = bt
                    st.session_state.ths = bs
                    st.session_state.tpr = pr
                except Exception as e:
                    sb.empty(); st.error(f"Humanizer error: {e}")

        if st.session_state.th:
            ht = st.session_state.th; hs = st.session_state.ths; pr = st.session_state.tpr
            _, _, nai, _ = text_verdict(hs)
            st.divider()
            st.subheader("📊 Before vs After")
            c1, c2, c3 = st.columns(3)
            c1.metric("Original AI Score",  f"{ai_pct:.1f}%")
            c2.metric("Humanized AI Score", f"{nai:.1f}%", delta=f"{nai-ai_pct:+.1f}%", delta_color="inverse")
            c3.metric("Improvement",        f"{ai_pct-nai:.1f}%")
            if len(pr) > 1:
                with st.expander(f"🔁 Pipeline ({len(pr)} passes)"):
                    for r in pr:
                        st.write(f"{'✅' if r['score']<=0.45 else '⚠️'} Pass {r['pass']}: **{r['score']*100:.1f}%**")
            if hs > 0.5: st.warning("Still reads as AI. Try editing a few sentences manually.")
            else:         st.success("Now reads as human-written. ✅")
            st.text_area("Humanized text", value=ht, height=280, key="to")
            with st.expander("📋 Side-by-side"):
                l, r = st.columns(2)
                l.markdown("**Original**")
                l.markdown(f"<div style='background:#0e0f1a;padding:10px;border-radius:8px;font-size:0.84em;line-height:1.6;max-height:380px;overflow-y:auto;'>{tin.replace(chr(10),'<br>')}</div>", unsafe_allow_html=True)
                r.markdown("**Humanized**")
                r.markdown(f"<div style='background:#0e0f1a;padding:10px;border-radius:8px;font-size:0.84em;line-height:1.6;max-height:380px;overflow-y:auto;'>{ht.replace(chr(10),'<br>')}</div>", unsafe_allow_html=True)

    st.markdown('<div class="info-box"><b>Models:</b> Bidirectional LSTM (Standard) · 5-fold CV LSTM (Pro)<br><b>Humanizer:</b> GPT-4o · up to 3 escalating passes with auto re-scoring</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — IMAGE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_image:
    import keras
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
    from PIL import Image as PILImage

    @st.cache_resource(show_spinner="Loading image models…")
    def load_image_models(ep, ap, cp):
        missing = [p for p in [ep, ap, cp] if not os.path.exists(p)]
        if missing: raise FileNotFoundError(f"Missing: {missing}")
        eff     = keras.models.load_model(ep, compile=False)
        eff_art = keras.models.load_model(ap, compile=False)
        return eff, eff_art, cp

    def _strategy():
        try:
            res = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(res)
            tf.tpu.experimental.initialize_tpu_system(res)
            return tf.distribute.TPUStrategy(res)
        except ValueError:
            return tf.distribute.get_strategy()

    def build_cnn(weights_path):
        m = Sequential([
            Conv2D(16,(3,3),strides=(1,1),activation='relu',input_shape=(256,256,3)),
            BatchNormalization(), MaxPooling2D(),
            Conv2D(32,(3,3),activation='relu'), BatchNormalization(), MaxPooling2D(),
            Conv2D(64,(3,3),activation='relu'), BatchNormalization(), MaxPooling2D(),
            Flatten(), Dense(512,activation='relu'), Dropout(0.09), Dense(1,activation='sigmoid'),
        ])
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        m.load_weights(weights_path)
        return m

    def img_run(pil_img, model_name, eff, eff_art, cnn_w):
        img = pil_img.convert("RGB")
        if model_name == "CNN":
            arr = np.array(img.resize((256,256)),dtype=np.float32)/255.0
            return build_cnn(cnn_w).predict(arr.reshape(1,256,256,3), verbose=0)
        elif model_name == "EfficientNet":
            arr = np.expand_dims(np.array(img.resize((300,300)),dtype=np.float32),0)
            with _strategy().scope(): return eff.predict(arr, verbose=0)
        elif model_name == "EfficientNet Art":
            arr = np.expand_dims(np.array(img.resize((224,224)),dtype=np.float32),0)
            with _strategy().scope(): return eff_art.predict(arr, verbose=0)

    def img_interpret(preds):
        s = float(preds[0][0])
        if s > 0.5: return "Real Photo",   s, int(s*100),      "real"
        else:       return "AI Generated", s, int((1-s)*100),  "ai"

    st.subheader("🖼️ AI Image Detector")
    st.caption("CNN · EfficientNetB3 · EfficientNet Art — each with its original input resolution.")
    st.divider()

    if "img_prev" not in st.session_state: st.session_state.img_prev = None
    if "img_mk"   not in st.session_state: st.session_state.img_mk   = "imk0"

    il, ir = st.columns([1,1], gap="large")
    with il:
        uimg = st.file_uploader("Upload image", type=["png","jpg","jpeg","webp"], key="iup")
        if uimg != st.session_state.img_prev:
            st.session_state.img_mk   = "imk1" if st.session_state.img_mk=="imk0" else "imk0"
            st.session_state.img_prev = uimg
        if uimg:
            ib = uimg.read()
            if len(ib)/(1024**2) > 20: st.error("Max 20 MB."); st.stop()
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


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — VIDEO  (ResNeXt50 + LSTM · face-aware · ensemble mode)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_video:
    import torch
    import torch.nn as nn
    import torch.nn.functional as VF
    from torchvision import models as tv_models, transforms as tv_transforms
    import cv2
    from PIL import Image as PILImage

    class _VideoModel(nn.Module):
        def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1,
                     hidden_dim=2048, bidirectional=False):
            super().__init__()
            backbone      = tv_models.resnext50_32x4d(weights=None)
            self.model    = nn.Sequential(*list(backbone.children())[:-2])
            self.lstm     = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
            self.relu     = nn.LeakyReLU()
            self.dp       = nn.Dropout(0.4)
            self.linear1  = nn.Linear(2048, num_classes)
            self.avgpool  = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            batch_size, seq_length, c, h, w = x.shape
            x      = x.view(batch_size * seq_length, c, h, w)
            fmap   = self.model(x)
            x      = self.avgpool(fmap)
            x      = x.view(batch_size, seq_length, 2048)
            x_lstm, _ = self.lstm(x, None)
            return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

    VID_IM_SIZE   = 112
    VID_TRANSFORM = tv_transforms.Compose([
        tv_transforms.ToPILImage(),
        tv_transforms.Resize((VID_IM_SIZE, VID_IM_SIZE)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    VID_MODELS_AVAILABLE = {
        "97% acc · 100 frames · FF"       : ("model_97_acc_100_frames_FF_data.pt",       100),
        "97% acc · 80 frames · FF"        : ("model_97_acc_80_frames_FF_data.pt",          80),
        "97% acc · 60 frames · FF"        : ("model_97_acc_60_frames_FF_data.pt",          60),
        "95% acc · 40 frames · FF"        : ("model_95_acc_40_frames_FF_data.pt",          40),
        "93% acc · 100 frames · Celeb+FF" : ("model_93_acc_100_frames_celeb_FF_data.pt", 100),
    }

    @st.cache_resource(show_spinner=False)
    def _load_vid_model(model_filename, model_dir):
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(model_dir, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Video model not found: {model_path}")
        m = _VideoModel(num_classes=2)
        m.load_state_dict(torch.load(model_path, map_location=device))
        m.to(device).eval()
        return m, device

    def _detect_and_crop_face(frame_bgr):
        gray     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces    = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            pad = int(0.1 * min(w, h))
            x1 = max(0, x-pad);             y1 = max(0, y-pad)
            x2 = min(frame_bgr.shape[1], x+w+pad)
            y2 = min(frame_bgr.shape[0], y+h+pad)
            return frame_bgr[y1:y2, x1:x2]
        return frame_bgr

    def _extract_vid_frames(video_path, sequence_length):
        cap          = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n            = min(sequence_length, total_frames)
        indices      = [int(i * total_frames / n) for i in range(n)]
        tensors, display_frames = [], []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue
            face = _detect_and_crop_face(frame)
            rgb  = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil  = PILImage.fromarray(rgb).resize((VID_IM_SIZE, VID_IM_SIZE))
            tensors.append(VID_TRANSFORM(rgb))
            display_frames.append((idx / fps, pil))
        cap.release()
        if not tensors: return None, []
        while len(tensors) < sequence_length: tensors.append(tensors[-1])
        stacked = torch.stack(tensors[:sequence_length])
        return stacked.unsqueeze(0), display_frames

    def _get_vid_info(video_path):
        cap     = cv2.VideoCapture(video_path)
        fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        dur     = total / fps
        dur_str = (f"{int(dur//60)}m {int(dur%60)}s" if dur >= 60 else f"{dur:.1f}s")
        return {"fps": fps, "total": total, "w": w, "h": h, "dur": dur, "dur_str": dur_str}

    def _run_vid_prediction(model, device, video_tensor):
        video_tensor = video_tensor.float().to(device)
        with torch.no_grad():
            fmap, logits = model(video_tensor)
            probs = VF.softmax(logits, dim=1)[0]
        return int(probs.argmax().item()), probs[0].item(), probs[1].item()

    st.subheader("🎬 Video Detector")
    st.caption("ResNeXt50 + LSTM · face-aware · sequence modeling · 5 pretrained models · ensemble mode.")
    st.divider()

    vl, vr = st.columns([1,1], gap="large")
    with vl:
        uvid = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv","webm"], key="vup")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label-vid">Model</div>', unsafe_allow_html=True)
        vid_model_choice = st.selectbox(
            "model", list(VID_MODELS_AVAILABLE.keys()), index=0,
            label_visibility="collapsed",
            help="Higher frame count = more temporal context = slower but more accurate.",
            key="vid_sel",
        )
        vid_model_filename, vid_seq_len = VID_MODELS_AVAILABLE[vid_model_choice]
        st.caption(f"Sequence {vid_seq_len} frames · `{vid_model_filename}`")
        st.markdown("<br>", unsafe_allow_html=True)
        vid_run_all = st.checkbox("Run all 5 models (ensemble)", value=False,
                                  help="Runs every model and shows individual + majority-vote verdict.")
        st.caption("Ensemble uses majority vote across all 5 models.")
        st.markdown("<br>", unsafe_allow_html=True)
        vid_analyze_btn = st.button("🎬 Analyze Video", type="primary", use_container_width=True, key="vbtn")
    with vr:
        vid_result_ph = st.empty()

    if uvid and not vid_analyze_btn:
        with vr: vid_result_ph.video(uvid)

    if uvid and vid_analyze_btn:
        vid_bytes = uvid.read()
        if len(vid_bytes)/(1024**2) > 500:
            st.error("File too large. Max 500 MB."); st.stop()
        suffix = os.path.splitext(uvid.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(vid_bytes); tmp_path = tmp.name
        try:
            info = _get_vid_info(tmp_path)
            models_to_run = (list(VID_MODELS_AVAILABLE.items()) if vid_run_all
                             else [(vid_model_choice, (vid_model_filename, vid_seq_len))])
            all_results = []; display_frames = []
            n_models = len(models_to_run)
            progress = st.progress(0); status = st.empty()

            for m_idx, (m_name, (m_file, m_seq)) in enumerate(models_to_run):
                status.info(f"[{m_idx+1}/{n_models}] Loading {m_name}…")
                try:    model_v, device_v = _load_vid_model(m_file, vid_model_dir)
                except FileNotFoundError as e: st.error(str(e)); st.stop()
                status.info(f"[{m_idx+1}/{n_models}] Extracting {m_seq} frames + face detection…")
                video_tensor, display_frames = _extract_vid_frames(tmp_path, m_seq)
                if video_tensor is None:
                    st.error("Could not extract frames. Try a different video format."); st.stop()
                status.info(f"[{m_idx+1}/{n_models}] Running inference…")
                pred, fake_p, real_p = _run_vid_prediction(model_v, device_v, video_tensor)
                all_results.append((m_name, pred, fake_p, real_p))
                progress.progress((m_idx+1)/n_models)

            progress.empty(); status.empty()

            fake_votes = sum(1 for _, p, _, _ in all_results if p == 0)
            real_votes = len(all_results) - fake_votes
            is_fake    = fake_votes >= real_votes if vid_run_all else (all_results[0][1] == 0)
            avg_fake   = float(np.mean([fp for _,_,fp,_ in all_results]))
            avg_real   = float(np.mean([rp for _,_,_,rp in all_results]))
            verdict    = "DEEPFAKE"  if is_fake else "AUTHENTIC"
            css_cls    = "fake"      if is_fake else "real"
            conf_pct   = int(avg_fake*100) if is_fake else int(avg_real*100)

            with vr:
                vid_result_ph.markdown(
                    result_card_html(
                        verdict, css_cls, conf_pct,
                        avg_fake if is_fake else avg_real,
                        (f'<b>{"Ensemble verdict" if vid_run_all else "Verdict"}</b>'
                         f'{"&nbsp;·&nbsp; Fake " + str(fake_votes) + "/" + str(len(all_results)) + " models" if vid_run_all else ""}'
                         f'<br><b>Fake score</b> · {avg_fake:.4f} &nbsp; <b>Real score</b> · {avg_real:.4f}<br>'
                         f'<b>Resolution</b> · {info["w"]}×{info["h"]} &nbsp;·&nbsp; <b>Duration</b> · {info["dur_str"]}<br>'
                         f'<b>FPS</b> · {info["fps"]:.1f} &nbsp;·&nbsp; <b>Total frames</b> · {info["total"]:,}<br>'
                         f'<b>Device</b> · {str(device_v).upper()} &nbsp;·&nbsp; <b>Model</b> · ResNeXt50+LSTM')
                    ), unsafe_allow_html=True)

            if vid_run_all:
                st.markdown('<div class="section-label-vid" style="margin-top:1.2rem;">Per-model breakdown</div>', unsafe_allow_html=True)
                cards_html = '<div class="model-cards">'
                for m_name, pred, fp, rp in all_results:
                    mc_cls = "fake" if pred==0 else "real"
                    mc_lbl = "FAKE" if pred==0 else "REAL"
                    mc_conf = int(fp*100) if pred==0 else int(rp*100)
                    short   = m_name.split("·")[0].strip()
                    cards_html += f'<div class="model-card"><div class="mc-name">{short}</div><div class="mc-score {mc_cls}">{mc_lbl}</div><div class="mc-conf">{mc_conf}% confidence</div></div>'
                cards_html += '</div>'
                st.markdown(cards_html, unsafe_allow_html=True)

            n_f = len(display_frames)
            segs = "".join(
                f'<div class="tl-seg tl-{"fake" if (i/max(n_f,1))<avg_fake else "real"}" title="frame {i}"></div>'
                for i in range(n_f))
            st.caption(f"Sequence timeline · {n_f} frames  🔴 Fake-leaning · 🟢 Real-leaning")
            st.markdown(f'<div class="tl-bar">{segs}</div>', unsafe_allow_html=True)

            st.caption("Sampled & face-cropped frames")
            CPR = 8
            for row in range(math.ceil(n_f/CPR)):
                cols = st.columns(CPR)
                for ci, col in enumerate(cols):
                    idx2 = row*CPR+ci
                    if idx2 >= n_f: break
                    ts, pil = display_frames[idx2]
                    with col:
                        st.image(pil, use_container_width=True)
                        st.markdown(f'<div class="frame-label {css_cls}">{verdict}<br><span style="opacity:0.5">{ts:.1f}s</span></div>', unsafe_allow_html=True)
        finally:
            if os.path.exists(tmp_path): os.unlink(tmp_path)

    st.markdown("""
<div class="info-box">
    <b>Architecture:</b> ResNeXt50_32x4d → AdaptiveAvgPool2d(1) → LSTM(2048) → mean pool → Dropout(0.4) → Linear(2048,2).
    Softmax: index 0 = FAKE · index 1 = REAL.<br><br>
    <b>Preprocessing:</b> Evenly sampled frames · Haar cascade face crop (fallback: full frame) · 112×112 · ImageNet normalisation.<br><br>
    <b>Models:</b> FaceForensics++ and Celeb-DF · 40/60/80/100 frame sequences · 97% best accuracy · auto-downloaded from Google Drive.
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — AUDIO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_audio:
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    AUD_MELS   = 128
    AUD_FRAMES = 87
    AUD_MAX_MB = 50
    AUD_MAX_S  = 60.0

    @st.cache_resource(show_spinner="Loading audio model…")
    def load_audio_model(path):
        if not os.path.exists(path): raise FileNotFoundError(f"Audio model not found: {path}")
        return keras.models.load_model(path)

    def audio_features(fp):
        audio, sr = librosa.load(fp, sr=None, mono=True)
        dur  = librosa.get_duration(y=audio, sr=sr)
        mel  = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=AUD_MELS)
        mdb  = librosa.power_to_db(mel, ref=np.max)
        mdb_f = (np.pad(mdb, ((0,0),(0,AUD_FRAMES-mdb.shape[1])), mode="reflect")
                 if mdb.shape[1] < AUD_FRAMES else mdb[:, :AUD_FRAMES])
        return mdb_f[np.newaxis,...], mdb_f, sr, dur

    def plot_wave(fp, sr):
        audio, _ = librosa.load(fp, sr=sr, mono=True)
        fig, ax  = plt.subplots(figsize=(8,2))
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        ax.plot(np.linspace(0, len(audio)/sr, len(audio)), audio, color="#a0a8ff", lw=0.6, alpha=0.9)
        ax.set_xlabel("Time (s)", fontsize=9); ax.set_ylabel("Amplitude", fontsize=9)
        ax.tick_params(labelsize=8); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); return fig

    def plot_mel(mdb, sr):
        fig, ax = plt.subplots(figsize=(8,3))
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        img = librosa.display.specshow(mdb, sr=sr, x_axis="frames", y_axis="mel", ax=ax, cmap="magma")
        fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
        ax.set_title("Mel Spectrogram (model input)", fontsize=10)
        ax.tick_params(labelsize=8); fig.tight_layout(); return fig

    st.subheader("🎙️ Audio Detector")
    st.caption("Mel spectrogram CNN · WAV, MP3, FLAC, OGG · reflect padding · waveform + spectrogram visualization.")
    st.divider()

    sw   = st.toggle("Show waveform",        value=True, key="asw")
    sm   = st.toggle("Show mel spectrogram", value=True, key="asm")
    uaud = st.file_uploader("Upload audio", type=["wav","mp3","flac","ogg"], key="aup")

    if uaud:
        ab  = uaud.read()
        amb = len(ab)/(1024**2)
        if amb > AUD_MAX_MB: st.error(f"File too large ({amb:.1f} MB). Max {AUD_MAX_MB} MB."); st.stop()
        st.audio(uaud)
        suf = "." + uaud.name.rsplit(".",1)[-1]
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
            conf_a = pred if fake_a else 1-pred
            lbl_a  = "FAKE" if fake_a else "REAL"
            cls_a  = "fake" if fake_a else "real"
            st.divider()
            rc, bc = st.columns([1,2])
            with rc:
                st.markdown(result_card_html(lbl_a, cls_a, int(conf_a*100), pred, f"<b>File</b> · {uaud.name}"), unsafe_allow_html=True)
            with bc:
                st.write("**Score breakdown**")
                st.progress(float(pred),   text=f"Fake: {pred:.1%}")
                st.progress(float(1-pred), text=f"Real: {1-pred:.1%}")
                certainty = ("Very high" if conf_a>0.90 else "High" if conf_a>0.75
                             else "Moderate" if conf_a>0.60 else "Low — borderline")
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



