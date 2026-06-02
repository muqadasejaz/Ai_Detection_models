# # ─────────────────────────────────────────────────────────────────────────────
# #  pages/2_🖼️_Image_Detection.py
# #  AI Image Detector — EfficientNetV2-B3
# #
# #  Training facts (ai-vs-real-images.ipynb):
# #    Input  : 224x224x3, float32 [0,255] — DO NOT /255
# #             include_preprocessing=True normalises inside backbone
# #    Output : sigmoid scalar — 0=Real, 1=AI Generated
# #    Rule   : score >= 0.5 -> AI Generated | score < 0.5 -> Real Photo
# #
# #  Secret:
# #    [gdrive]
# #    cnn_detection = "GOOGLE_DRIVE_FILE_ID"
# # ─────────────────────────────────────────────────────────────────────────────
# import os
# import io

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import numpy as np
# import streamlit as st
# from PIL import Image as PILImage

# st.set_page_config(
#     page_title="Image Detection · DeepSentinel",
#     page_icon="🖼️",
#     layout="wide",
# )

# from utils import inject_css, result_card_html, ensure_file, load_keras_model
# inject_css()

# IMG_SIZE       = (224, 224)
# MODEL_FILENAME = "cnn_detection.h5"


# # ─────────────────────────────────────────────────────────────────────────────
# #  MODEL — downloaded once (via ensure_file), loaded once, reused forever.
# #  @st.cache_resource keeps the model object in memory for the app lifetime.
# #
# #  FIX: get_model() is NOT called at module level. It is called only inside
# #  the upload handler below. This prevents TensorFlow + Keras from
# #  initialising (and consuming ~1 GB RAM) before any user uploads an image,
# #  which was causing an OOM → segfault on Streamlit Cloud's free tier.
# # ─────────────────────────────────────────────────────────────────────────────
# @st.cache_resource(show_spinner="Loading EfficientNet model... (first visit only)")
# def get_model():
#     path = ensure_file(MODEL_FILENAME)
#     return load_keras_model(path, compile=False)


# # ─────────────────────────────────────────────────────────────────────────────
# #  INFERENCE HELPERS
# # ─────────────────────────────────────────────────────────────────────────────
# def preprocess(pil_img: PILImage.Image) -> np.ndarray:
#     """
#     Matches training load_image() exactly:
#       resize to 224x224, cast float32, keep [0,255].
#     DO NOT divide by 255 — backbone uses include_preprocessing=True.
#     """
#     img = pil_img.convert("RGB").resize(IMG_SIZE, PILImage.LANCZOS)
#     arr = np.array(img, dtype=np.float32)   # [0.0 .. 255.0]
#     return np.expand_dims(arr, axis=0)      # (1, 224, 224, 3)


# def predict(model, pil_img: PILImage.Image) -> float:
#     return float(model.predict(preprocess(pil_img), verbose=0)[0][0])


# def interpret(score: float):
#     if score >= 0.5:
#         return "AI Generated", "ai",   int(round(score * 100))
#     return "Real Photo",       "real", int(round((1.0 - score) * 100))


# # ─────────────────────────────────────────────────────────────────────────────
# #  UI
# # ─────────────────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("## 🛡️ DeepSentinel")
#     st.caption("AI & Deepfake Detection Suite")

# st.markdown('<div class="hero-title">🖼️ AI Image Detector</div>', unsafe_allow_html=True)
# st.divider()

# if "img_prev" not in st.session_state:
#     st.session_state.img_prev = None

# col_left, col_right = st.columns([1, 1], gap="large")

# with col_left:
#     uploaded = st.file_uploader(
#         "Upload an image",
#         type=["png", "jpg", "jpeg", "webp", "bmp"],
#         key="img_uploader",
#     )
#     if uploaded != st.session_state.img_prev:
#         st.session_state.img_prev = uploaded
#     raw = None
#     if uploaded:
#         raw = uploaded.read()
#         if len(raw) / (1024 ** 2) > 20:
#             st.error("Max 20 MB.")
#             st.stop()
#         st.image(raw, use_container_width=True)

# with col_right:
#     result_slot = st.empty()

#     if uploaded and raw is not None:
#         with st.spinner("Analysing image..."):
#             try:
#                 # FIX: model is loaded here (lazily) not at module level.
#                 # @st.cache_resource ensures it only loads once per app lifetime
#                 # and is reused on every subsequent upload — no performance cost.
#                 model = get_model()
#                 pil   = PILImage.open(io.BytesIO(raw))
#                 score = predict(model, pil)
#             except Exception as exc:
#                 result_slot.error(f"Inference failed: {exc}")
#                 st.stop()

#         label, css_cls, conf_pct = interpret(score)
#         result_slot.markdown(
#             result_card_html(
#                 label, css_cls, conf_pct, score,
#                 f"<b>Model</b> · EfficientNet &nbsp;·&nbsp; <b>File</b> · {uploaded.name}",
#             ),
#             unsafe_allow_html=True,
#         )



}

"""
AI Image Detector — Phase 1 (metadata) + Phase 2 (CNN visual pass)
"""

import io, struct, zlib, re, json, os, warnings
from pathlib import Path

import streamlit as st
from PIL import Image, ExifTags
import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────
# SHARED CSS + Keras loader from utils
# ─────────────────────────────────────────────────────────────────
from utils import inject_css, load_keras_model, ensure_file   # ← added
inject_css()                                      # ← added

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d0d0d; color: #e8e8e8; }

[data-testid="stFileUploader"] {
    border: 2px dashed #2a2a2a;
    border-radius: 16px;
    padding: 24px;
    background: #141414;
}

/* ── Verdict banner — AI found (green pill like Vynly) ── */
.verdict-ai {
    display: inline-flex; align-items: center; gap: 10px;
    background: #052e16; border: 1px solid #166534;
    border-radius: 999px; padding: 9px 20px; margin: 20px 0 0;
}
.verdict-ai .check { color: #4ade80; font-size: 15px; }
.verdict-ai .title { color: #4ade80; font-weight: 600; font-size: 14px; }
.verdict-ai .sub {
    color: #6b7280; font-size: 13px; margin-left: 6px;
    padding-left: 10px; border-left: 1px solid #166534;
}

/* ── Verdict banner — clean (grey) ── */
.verdict-clean {
    display: inline-flex; align-items: center; gap: 10px;
    background: #141414; border: 1px solid #2a2a2a;
    border-radius: 999px; padding: 9px 20px; margin: 20px 0 0;
}
.verdict-clean .title { color: #6b7280; font-weight: 600; font-size: 14px; }

/* ── Verdict banner — model prediction (blue tint) ── */
.verdict-model {
    display: inline-flex; align-items: center; gap: 10px;
    background: #0a1628; border: 1px solid #1e3a5f;
    border-radius: 999px; padding: 9px 20px; margin: 10px 0 0;
}
.verdict-model .icon { color: #7dd3fc; font-size: 15px; }
.verdict-model .title { color: #7dd3fc; font-weight: 600; font-size: 14px; }
.verdict-model .sub {
    color: #6b7280; font-size: 13px; margin-left: 6px;
    padding-left: 10px; border-left: 1px solid #1e3a5f;
}

/* ── Evidence block ── */
.evidence-header {
    font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
    color: #4b5563; text-transform: uppercase;
    margin: 18px 0 8px; padding: 0;
}
.ev-row {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 0; border-bottom: 1px solid #1a1a1a;
}
.ev-row:last-child { border-bottom: none; }
.ev-tag {
    font-size: 9px; font-weight: 700; letter-spacing: 0.07em;
    padding: 3px 8px; border-radius: 4px; flex-shrink: 0;
    text-transform: uppercase;
}
.tag-bytes { background:#0f172a; color:#93c5fd; border:1px solid #1e3a8a; }
.tag-exif  { background:#1e1b4b; color:#c4b5fd; border:1px solid #3730a3; }
.tag-xmp   { background:#042f2e; color:#5eead4; border:1px solid #134e4a; }
.tag-c2pa  { background:#1c1007; color:#fdba74; border:1px solid #92400e; }
.tag-cnn   { background:#082f49; color:#7dd3fc; border:1px solid #075985; }
.ev-text {
    font-size: 12.5px; color: #9ca3af;
    background: #1a1a1a; border: 1px solid #272727;
    border-radius: 5px; padding: 2px 8px;
    font-family: 'Courier New', monospace;
}

/* ── Vision pass clean items ── */
.vp-subline { font-size: 13px; color: #4b5563; margin: 14px 0 10px; }
.vp-block { margin: 4px 0 16px; padding: 0; }
.vp-item {
    font-size: 13px; color: #374151; padding: 3px 0;
    display: flex; align-items: center; gap: 8px;
}
.vp-item::before { content: "·"; color: #374151; font-size: 20px; line-height:1; flex-shrink:0; }

/* ── "What this does NOT mean" block ── */
.notmean-block {
    background: #0d0d0d; border: 1px solid #1e1e1e;
    border-left: 3px solid #27272a;
    border-radius: 8px; padding: 18px 22px; margin: 4px 0 0;
    font-size: 13px; color: #4b5563; line-height: 1.75;
}
.notmean-block strong {
    color: #6b7280; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em;
    display: block; margin-bottom: 10px;
}

/* ── File meta ── */
.file-meta {
    text-align: center; font-size: 12px; color: #374151;
    margin-top: 6px;
}

/* ── Info box ── */
.info-box {
    background: #0f0f0f; border: 1px solid #1a1a1a;
    border-radius: 12px; padding: 20px 24px; margin-top: 32px;
    font-size: 13px; color: #4b5563; line-height: 1.75;
}
.info-box h4 { color: #6b7280; font-size: 12px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; margin: 0 0 14px; }
.info-box ul { margin: 0; padding-left: 18px; }
.info-box li { margin-bottom: 8px; }
.info-box code { background:#1a1a1a; padding:1px 5px; border-radius:4px;
    font-size:12px; color:#9ca3af; }
.info-box p { margin: 12px 0 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
KNOWN_AI_SOFTWARE = [
    "midjourney", "stable diffusion", "flux", "dall-e", "dall-e", "sora",
    "firefly", "ideogram", "leonardo", "runway", "imagen", "comfyui",
    "a1111", "automatic1111", "novelai", "dreamshaper", "kandinsky",
    "playground ai", "blue willow", "gencraft", "artbreeder",
    "nightcafe", "adobe firefly", "bing image creator", "adobe ai",
    "stability ai", "getimg", "invoke ai", "diffusers", "gemini",
]

AI_XMP_VALUES = [
    "trainedalgorithmicmedia", "compositewithtrainedalgorithmicmedia",
    "algorithmicmedia", "compositewithalgorithmicmedia",
    "softwareagent", "trainedmodel",
]

C2PA_SIGNATURES = [b"c2pa", b"jumb", b"JUMB", b"jumd"]
C2PA_MARKERS    = [b"c2pa.org", b"contentauthenticity", b"cai.adobe.com",
                   b"c2patool", b"TrustedAlgorithmicMedia"]

Evidence = dict


# ─────────────────────────────────────────────────────────────────
# PHASE 1 — BYTE-LEVEL METADATA
# ─────────────────────────────────────────────────────────────────

def check_c2pa(raw: bytes):
    hits = []
    for sig in C2PA_SIGNATURES:
        if sig in raw:
            hits.append({"tag":"BYTES","tag_class":"tag-bytes",
                "label":"Cryptographic provenance manifest",
                "text":"C2PA chunk present in PNG"}); break
    for m in C2PA_MARKERS:
        if m in raw:
            hits.append({"tag":"C2PA","tag_class":"tag-c2pa",
                "label":"Content credentials",
                "text":f"Content credentials: {m.decode(errors='replace')}"}); break
    return hits


def check_png_chunks(raw: bytes, fmt: str):
    hits = []
    if fmt != "PNG" or not raw.startswith(b"\x89PNG"):
        return hits
    pos = 8
    while pos < len(raw) - 12:
        try:
            length = struct.unpack(">I", raw[pos:pos+4])[0]
            ctype  = raw[pos+4:pos+8].decode("ascii", errors="replace")
            data   = raw[pos+8:pos+8+length]
            if ctype in ("tEXt","iTXt","zTXt"):
                try:
                    text = data.decode("utf-8", errors="replace") if ctype != "zTXt" else ""
                    if ctype == "zTXt":
                        null = data.find(b"\x00")
                        if null > 0:
                            text = zlib.decompress(data[null+2:]).decode("utf-8", errors="replace")
                except Exception:
                    text = ""
                tl = text.lower()
                if "parameters" in tl or "negative prompt" in tl:
                    snippet = text[:80].replace("\n"," ").strip()
                    hits.append({"tag":"BYTES","tag_class":"tag-bytes",
                        "label":f"PNG {ctype} — A1111 parameters",
                        "text":f"A1111 parameters key: {snippet[:60]}..."})
                if "workflow" in tl and ("comfyui" in tl or "sampler" in tl):
                    hits.append({"tag":"BYTES","tag_class":"tag-bytes",
                        "label":f"PNG {ctype} — ComfyUI workflow",
                        "text":"ComfyUI workflow JSON in PNG metadata"})
                if "c2pa" in tl or "contentcredentials" in tl:
                    hits.append({"tag":"C2PA","tag_class":"tag-c2pa",
                        "label":f"PNG {ctype} — C2PA",
                        "text":"C2PA chunk present in PNG"})
                for av in AI_XMP_VALUES:
                    if av in tl:
                        hits.append({"tag":"BYTES","tag_class":"tag-bytes",
                            "label":"XMP DigitalSourceType",
                            "text":f"XMP DigitalSourceType = {av}"}); break
                for sw in KNOWN_AI_SOFTWARE:
                    if sw in tl:
                        hits.append({"tag":"BYTES","tag_class":"tag-bytes",
                            "label":f"PNG {ctype} — generator tag",
                            "text":f"Metadata mentions {sw.title()}"}); break
            pos += 12 + length
        except Exception:
            break
    return hits


def check_exif(img: Image.Image):
    hits = []
    try:
        exif_data = img._getexif()
        if not exif_data:
            return hits
        for tag_id, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if not isinstance(value, str):
                continue
            vl = value.lower()
            if tag_name in ("Software","ProcessingSoftware","Make","Model",
                            "Artist","ImageDescription","UserComment"):
                for sw in KNOWN_AI_SOFTWARE:
                    if sw in vl:
                        hits.append({"tag":"EXIF","tag_class":"tag-exif",
                            "label":f"EXIF {tag_name}",
                            "text":f"{tag_name} = {value[:100]}"}); break
    except Exception:
        pass
    return hits


def check_xmp_raw(raw: bytes):
    hits = []
    try:
        start = raw.find(b"<x:xmpmeta")
        if start == -1:
            start = raw.find(b"<?xpacket")
        if start == -1:
            return hits
        end  = raw.find(b"</x:xmpmeta>", start)
        blob = raw[start: end+12 if end != -1 else start+8192].decode("utf-8", errors="replace").lower()
        for av in AI_XMP_VALUES:
            if av in blob:
                hits.append({"tag":"BYTES","tag_class":"tag-bytes",
                    "label":"XMP DigitalSourceType",
                    "text":f"XMP DigitalSourceType = {av}"}); break
        if "synthid" in blob:
            hits.append({"tag":"XMP","tag_class":"tag-xmp",
                "label":"XMP — SynthID","text":"SynthID watermark tag in XMP"})
        for sw in KNOWN_AI_SOFTWARE:
            if sw in blob:
                hits.append({"tag":"XMP","tag_class":"tag-xmp",
                    "label":"XMP Software",
                    "text":f"Metadata mentions {sw.title()}"}); break
        m = re.search(r'creatortool[^>]*>([^<]{1,100})', blob)
        if m:
            creator = m.group(1).strip()
            for sw in KNOWN_AI_SOFTWARE:
                if sw in creator.lower():
                    hits.append({"tag":"XMP","tag_class":"tag-xmp",
                        "label":"XMP xmp:CreatorTool",
                        "text":f"xmp:CreatorTool = {creator}"}); break
    except Exception:
        pass
    return hits


def check_iptc(raw: bytes):
    hits = []
    try:
        idx = raw.find(b"\x1c\x02")
        if idx == -1:
            return hits
        blob = raw[idx:idx+2048].decode("latin-1", errors="replace").lower()
        for sw in KNOWN_AI_SOFTWARE:
            if sw in blob:
                hits.append({"tag":"XMP","tag_class":"tag-xmp",
                    "label":"IPTC metadata",
                    "text":f"Metadata mentions {sw.title()}"}); break
        for av in ["trainedalgorithmicmedia","algorithmicmedia"]:
            if av in blob:
                hits.append({"tag":"XMP","tag_class":"tag-xmp",
                    "label":"IPTC DigitalSourceType",
                    "text":f"IPTC DigitalSourceType = {av}"}); break
    except Exception:
        pass
    return hits


def run_phase1(img, raw, fmt):
    ev = []
    ev += check_c2pa(raw)
    ev += check_png_chunks(raw, fmt)
    ev += check_xmp_raw(raw)
    ev += check_exif(img)
    ev += check_iptc(raw)
    seen, out = set(), []
    for e in ev:
        k = e["text"][:60]
        if k not in seen:
            seen.add(k); out.append(e)
    return out


# ─────────────────────────────────────────────────────────────────
# PHASE 2 — CNN VISUAL PASS
# ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "cnn_detection.h5"

@st.cache_resource(show_spinner="Loading CNN model...")
def load_model():
    # ensure_file() downloads cnn_detection.h5 from Google Drive on first use.
    # load_keras_model() then handles the Keras 2/3 version mismatch.
    ensure_file("cnn_detection.h5")
    return load_keras_model(str(MODEL_PATH), compile=False)


def run_phase2(img: Image.Image):
    """
    Matches notebook inference exactly:
      img.convert(RGB).resize((224,224), LANCZOS) → float32 [0,255] → predict
      prob >= 0.5  →  AI-Generated
      prob <  0.5  →  Real
    DO NOT divide by 255 — EfficientNetV2B3 include_preprocessing=True
    handles normalisation internally.
    """
    if not MODEL_PATH.exists():
        return {"prob": None, "label": "unavailable", "conf_pct": None,
                "details": ["model file cnn_detection.h5 not found"]}
    try:
        model = load_model()
        arr  = np.expand_dims(
            np.array(img.convert("RGB").resize((224, 224), Image.LANCZOS), dtype=np.float32),
            axis=0,
        )  # shape (1,224,224,3), values [0,255] — no /255
        prob = float(model.predict(arr, verbose=0)[0][0])

        # Exactly as notebook: >= 0.5 → FAKE(AI), < 0.5 → REAL
        if prob >= 0.5:
            label    = "AI-Generated"
            conf_pct = int(round(prob * 100))
        else:
            label    = "Real"
            conf_pct = int(round((1.0 - prob) * 100))

        return {"prob": prob, "label": label, "conf_pct": conf_pct, "details": []}
    except Exception as e:
        return {"prob": None, "label": "error", "conf_pct": None,
                "details": [str(e)]}


# ─────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────

def render_evidence_rows(hits):
    rows = ""
    for ev in hits:
        rows += (
            f'<div class="ev-row">'
            f'<span class="ev-tag {ev["tag_class"]}">{ev["tag"]}</span>'
            f'<span class="ev-text">{ev["text"]}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div class="evidence-header">Evidence found</div>'
        f'<div style="background:#0d0d0d;border:1px solid #1a1a1a;border-radius:10px;padding:4px 16px">'
        f'{rows}</div>',
        unsafe_allow_html=True,
    )


def render_ai_verdict(hits):
    st.markdown(
        '<div class="verdict-ai">'
        '<span class="check">✓</span>'
        '<span class="title">AI provenance verified</span>'
        '<span class="sub">C2PA-signed</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    render_evidence_rows(hits)


def render_clean_verdict(p2):
    prob     = p2.get("prob")
    conf_pct = p2.get("conf_pct")
    label    = p2.get("label", "")

    # ── Pill 1: No Provenance Metadata ───────────────────────────
    st.markdown(
        '<div class="verdict-clean">'
        '<span class="title">No provenance metadata</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Pill 2: Model prediction ──────────────────────────────────
    if prob is None:
        st.markdown(
            '<div class="verdict-model">'
            '<span class="icon">◦</span>'
            '<span class="title">Model predict: unavailable</span>'
            '<span class="sub">place cnn_detection.h5 next to app.py</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        raw_str  = f"raw score: {prob:.4f}"
        if label == "AI-Generated":
            sub_str = f"P(AI-Generated) = {conf_pct}% · {raw_str}"
        else:
            sub_str = f"P(Real) = {conf_pct}% · {raw_str}"

        st.markdown(
            f'<div class="verdict-model">'
            f'<span class="icon">◎</span>'
            f'<span class="title">Model predict: {label}</span>'
            f'<span class="sub">{sub_str}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Disclaimer ───────────────────────────────────────────────
    st.markdown(
        '<div class="notmean-block">'
        '<strong>What this does NOT mean</strong>'
        '"No provenance metadata" does NOT mean the image is real / human-made. '
        'It means the file we inspected has no C2PA signature, SynthID watermark, '
        'XMP digital-source tag, or generator EXIF tag in its bytes. That happens to '
        'AI images all the time: most social media platforms, news-site CDNs, image '
        'proxies, screenshots, and re-encodes strip metadata. If the original generator '
        'embedded provenance, it&#39;s almost certainly gone by the time the image lands '
        'in your feed.<br><br>'
        'The model prediction is a heuristic signal, not a cryptographic proof. '
        'weight it accordingly.'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center;padding:40px 0 28px">
  <div style="font-size:34px;font-weight:700;letter-spacing:-1px;color:#f0f0f0">
    🔍 AI Image Detector
  </div>
  <div style="font-size:14px;color:#4b5563;margin-top:8px">
    cryptographic metadata , visual classifier
  </div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "", type=["jpg","jpeg","png","webp","avif","heic"],
    label_visibility="collapsed",
)
st.markdown(
    '<div style="text-align:center;font-size:12px;color:#2d2d2d;margin-top:-8px">'
    'JPEG · PNG · WebP · AVIF · HEIC — Max 25 MB</div>',
    unsafe_allow_html=True,
)

if uploaded:
    uploaded.seek(0)
    raw = uploaded.read()
    img = Image.open(io.BytesIO(raw))
    fmt = img.format or "UNKNOWN"
    size_kb = len(raw) / 1024

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        st.image(img, use_column_width=True)

    st.markdown(
        f'<div style="text-align:center;font-size:12px;color:#374151;margin-top:6px">'
        f'{size_kb:.1f} KB · {fmt.lower()} · {img.width}×{img.height}px</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    with st.spinner("Scanning metadata..."):
        p1_hits = run_phase1(img, raw, fmt)

    if p1_hits:
        render_ai_verdict(p1_hits)
        st.markdown("""
<div class="info-box">
<h4>What this checks</h4>
<ul>
  <li><strong>C2PA</strong> — Cryptographically signed manifests embedded by OpenAI (DALL-E, Sora),
      Adobe Firefly, Microsoft Designer, Google.</li>
  <li><strong>SynthID-style XMP/IPTC tags</strong> — <code>trainedAlgorithmicMedia</code>
      and related <code>DigitalSourceType</code> values.</li>
  <li><strong>EXIF / XMP Software tags</strong> — Midjourney, Flux, Ideogram, Leonardo,
      Runway, Imagen, Stable Diffusion, ComfyUI, A1111.</li>
  <li><strong>PNG text chunks</strong> — A1111's <code>parameters</code> key,
      ComfyUI's <code>workflow</code> JSON.</li>
  <li><strong>Visible watermarks</strong> (vision pass, runs only when byte-level pass finds nothing)
      — Gemini's corner mark, DALL-E rainbow corner, Sora / Midjourney / Imagen / Firefly
      content-credentials icon, and any visible "Made with AI" / generator overlay text.</li>
</ul>
<p>Two passes. The metadata pass is cryptographic — if it returns a hit, we can stand behind
it. The visual pass uses an AI vision model and is heuristic — we surface its confidence
(low / medium / high) so you can weight it accordingly. We deliberately do not ship a
vibes-based pixel classifier that confidently mislabels human work as AI; if we can't see
a watermark or read metadata, we say so.</p>
</div>
""", unsafe_allow_html=True)

    else:
        with st.spinner("Running visual classifier..."):
            p2 = run_phase2(img)
        render_clean_verdict(p2)
