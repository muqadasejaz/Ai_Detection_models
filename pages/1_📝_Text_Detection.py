# ─────────────────────────────────────────────────────────────────────────────
#  pages/1_📝_Text_Detection.py
#  AI Text Detector + GPT-4o Humanizer
#  Only tensorflow/keras loads when this page is visited.
# ─────────────────────────────────────────────────────────────────────────────
import os
import pickle
import streamlit as st

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

st.set_page_config(page_title="Text Detection · DeepSentinel", page_icon="📝", layout="wide")

from utils import inject_css, result_card_html
inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ DeepSentinel")
    st.caption("AI & Deepfake Detection Suite")
    st.divider()
    st.markdown("### 📁 Model Paths")
    text_model_standard = st.text_input("Standard model (.h5)",      value=os.environ.get("TEXT_MODEL_STANDARD", "lstm_main.h5"))
    text_tok_standard   = st.text_input("Standard tokenizer (.pkl)", value=os.environ.get("TEXT_TOK_STANDARD",   "tokenizer.pkl"))
    text_model_pro      = st.text_input("Pro model (.h5)",           value=os.environ.get("TEXT_MODEL_PRO",      "lstm_kfold.h5"))
    text_tok_pro        = st.text_input("Pro tokenizer (.pkl)",      value=os.environ.get("TEXT_TOK_PRO",        "tokenizer_lstm_best_kfold.pkl"))
    st.divider()
    st.markdown("### 🔑 OpenAI API Key")
    openai_api_key = st.text_input("For Text Humanizer", type="password",
                                   placeholder="sk-...", value=os.environ.get("OPENAI_API_KEY", ""))
    if openai_api_key:
        st.success("Humanizer ready ✅")

# ── Constants ─────────────────────────────────────────────────────────────────
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

# ── Model loaders ─────────────────────────────────────────────────────────────
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
    if prob > 0.75:   v, c = "🤖 Likely AI-Generated",    "#e74c3c"
    elif prob > 0.5:  v, c = "🤖 Possibly AI-Generated",  "#e67e22"
    elif prob > 0.25: v, c = "✍️ Possibly Human-Written", "#2980b9"
    else:             v, c = "✍️ Likely Human-Written",    "#27ae60"
    return v, c, ai, hu

def _openai_call(client, prompt, text, temp):
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt},
                  {"role": "user",   "content": f"Rewrite this text:\n\n{text}"}],
        temperature=temp, top_p=0.95, frequency_penalty=0.5, presence_penalty=0.4,
    )
    return r.choices[0].message.content.strip()

def humanize(text, api_key, model, tokenizer, threshold=0.45, aggressiveness="Balanced", status_cb=None):
    from openai import OpenAI
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    client = OpenAI(api_key=api_key)
    temps  = [0.88, 0.92, 0.95]
    cfg    = {"Light": {"start": 0, "max": 2}, "Balanced": {"start": 0, "max": 3}, "Aggressive": {"start": 1, "max": 3}}
    s_idx  = cfg[aggressiveness]["start"]
    mx     = cfg[aggressiveness]["max"]
    def score(t):
        seq = tokenizer.texts_to_sequences([t])
        pad = pad_sequences(seq, maxlen=TEXT_MAX_LEN, padding="post", truncating="post")
        return float(model.predict(pad, verbose=0)[0][0])
    best_t, best_s = text, score(text)
    cur = text; results = []
    for i in range(s_idx, s_idx + mx):
        idx = min(i, len(H_PROMPTS) - 1)
        pn  = i - s_idx + 1
        if status_cb: status_cb(f"Pass {pn}: {['light','structural','deep'][idx]} rewrite…")
        rw = _openai_call(client, H_PROMPTS[idx], cur, temps[idx])
        ns = score(rw)
        results.append({"pass": pn, "score": ns, "text": rw})
        if ns < best_s: best_s, best_t = ns, rw
        if ns <= threshold: break
        cur = rw
    return best_t, best_s, results

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("tp", None), ("tl", None), ("th", None), ("ths", None), ("tpr", [])]:
    if k not in st.session_state: st.session_state[k] = v

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📝 AI Text Detector</div>', unsafe_allow_html=True)
st.caption("Detect AI-generated text, then optionally rewrite it with GPT-4o.")
st.divider()

sel = st.radio("Detection model", list(TEXT_MODELS.keys()), index=0, horizontal=True)
mi  = TEXT_MODELS[sel]
st.caption(f"ℹ️ {mi['description']}")
st.divider()

tin = st.text_area("Paste your text here", height=200,
                   placeholder="Type or paste text to analyze…", max_chars=5000)
wc  = len(tin.split()) if tin.strip() else 0
st.caption(f"{wc} words · {len(tin)} characters")

if st.button("🔍 Analyze Text", type="primary", use_container_width=True, key="ta"):
    if not tin.strip():
        st.warning("Enter some text first.")
    elif wc < 10:
        st.warning("Paste at least 10 words for a reliable result.")
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
        ca, cb = st.columns([2, 1])
        with ca:
            agg = st.select_slider("Rewrite strength", ["Light", "Balanced", "Aggressive"], value="Balanced")
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
        c2.metric("Humanized AI Score", f"{nai:.1f}%", delta=f"{nai - ai_pct:+.1f}%", delta_color="inverse")
        c3.metric("Improvement",        f"{ai_pct - nai:.1f}%")
        if len(pr) > 1:
            with st.expander(f"🔁 Pipeline ({len(pr)} passes)"):
                for r in pr:
                    st.write(f"{'✅' if r['score'] <= 0.45 else '⚠️'} Pass {r['pass']}: **{r['score']*100:.1f}%**")
        if hs > 0.5: st.warning("Still reads as AI. Try editing a few sentences manually.")
        else:        st.success("Now reads as human-written. ✅")
        st.text_area("Humanized text", value=ht, height=280, key="to")
        with st.expander("📋 Side-by-side"):
            l, r = st.columns(2)
            l.markdown("**Original**")
            l.markdown(f"<div style='background:#0e0f1a;padding:10px;border-radius:8px;font-size:0.84em;line-height:1.6;max-height:380px;overflow-y:auto;'>{tin.replace(chr(10),'<br>')}</div>", unsafe_allow_html=True)
            r.markdown("**Humanized**")
            r.markdown(f"<div style='background:#0e0f1a;padding:10px;border-radius:8px;font-size:0.84em;line-height:1.6;max-height:380px;overflow-y:auto;'>{ht.replace(chr(10),'<br>')}</div>", unsafe_allow_html=True)

st.markdown('<div class="info-box"><b>Models:</b> Bidirectional LSTM (Standard) · 5-fold CV LSTM (Pro)<br><b>Humanizer:</b> GPT-4o · up to 3 escalating passes with auto re-scoring</div>', unsafe_allow_html=True)
