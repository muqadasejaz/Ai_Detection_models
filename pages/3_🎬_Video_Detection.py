# # ─────────────────────────────────────────────────────────────────────────────
# #  pages/3_🎬_Video_Detection.py
# #  Deepfake Video Detector — ResNeXt50 + LSTM · face-aware · ensemble
# #  Only torch/torchvision/cv2 loads when this page is visited.
# # ─────────────────────────────────────────────────────────────────────────────
# import os
# import math
# import tempfile
# import numpy as np
# import streamlit as st
# from PIL import Image as PILImage

# st.set_page_config(page_title="Video Detection · DeepSentinel", page_icon="🎬", layout="wide")

# from utils import inject_css, result_card_html, ensure_file
# inject_css()

# import torch
# import torch.nn as nn
# import torch.nn.functional as VF
# from torchvision import models as tv_models
# import cv2

# # ── Sidebar ───────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("## 🛡️ DeepSentinel")
#     st.caption("AI & Deepfake Detection Suite")
#     st.divider()
#     st.markdown("### 📁 Model Directory")
#     vid_model_dir = st.text_input(
#         "Model directory", value=os.environ.get("VID_MODEL_DIR", "."),
#         help="Informational only — ensure_file() controls the actual download/load path, "
#              "so this field no longer needs to match where files are saved.",
#     )

# # ── Model definition ──────────────────────────────────────────────────────────
# class _VideoModel(nn.Module):
#     def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1,
#                  hidden_dim=2048, bidirectional=False):
#         super().__init__()
#         backbone   = tv_models.resnext50_32x4d(weights=None)
#         self.model = nn.Sequential(*list(backbone.children())[:-2])
#         self.lstm  = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
#         self.relu  = nn.LeakyReLU()
#         self.dp    = nn.Dropout(0.4)
#         self.linear1 = nn.Linear(2048, num_classes)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)

#     def forward(self, x):
#         batch_size, seq_length, c, h, w = x.shape
#         x = x.view(batch_size * seq_length, c, h, w)
#         fmap = self.model(x)
#         x    = self.avgpool(fmap)
#         x    = x.view(batch_size, seq_length, 2048)
#         x_lstm, _ = self.lstm(x, None)
#         return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

# VID_IM_SIZE = 112

# # FIX: torchvision 0.16 ToTensor() crashes on certain PIL modes via mode_to_nptype.
# # We bypass torchvision entirely and convert manually — fully version-safe.
# _NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
# _NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# def _pil_to_tensor(pil_img):
#     """Version-safe PIL Image → normalised float32 tensor (3, H, W).
#     Replaces torchvision ToTensor() + Normalize() to avoid mode_to_nptype crash."""
#     arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0  # H,W,3  [0,1]
#     t   = torch.from_numpy(arr).permute(2, 0, 1)                       # 3,H,W
#     return (t - _NORM_MEAN) / _NORM_STD                                # normalised

# VID_MODELS_AVAILABLE = {
#     "97% acc · 100 frames · FF"       : ("model_97_acc_100_frames_FF_data.pt",        100),
#     "97% acc · 80 frames · FF"        : ("model_97_acc_80_frames_FF_data.pt",           80),
#     "97% acc · 60 frames · FF"        : ("model_97_acc_60_frames_FF_data.pt",           60),
#     "95% acc · 40 frames · FF"        : ("model_95_acc_40_frames_FF_data.pt",           40),
#     "93% acc · 100 frames · Celeb+FF" : ("model_93_acc_100_frames_celeb_FF_data.pt",  100),
# }

# # ── Model loaders & helpers ───────────────────────────────────────────────────
# @st.cache_resource(show_spinner=False)
# def _load_vid_model(model_filename):
#     """
#     Loads and caches a single video model, keyed on model_filename.
#     Because @st.cache_resource keys on arguments, each of the 5 models below
#     gets its own permanent cache entry — every model downloads and loads AT
#     MOST ONCE per app lifetime, whether called once or repeatedly via the
#     ensemble loop.

#     FIX: previously this re-joined model_dir + model_filename to build
#     model_path, but ensure_file() always downloads to the current working
#     directory regardless of model_dir. If someone changed the "Model
#     directory" sidebar field away from the default ".", torch.load() would
#     look in the wrong place and throw FileNotFoundError even though the file
#     existed on disk. Now we just trust the path ensure_file() returns.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_path = ensure_file(model_filename)   # downloads once, returns actual on-disk path
#     m = _VideoModel(num_classes=2)
#     m.load_state_dict(torch.load(model_path, map_location=device))
#     m.to(device).eval()
#     return m, device

# def _detect_and_crop_face(frame_bgr):
#     gray     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#     detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#     faces    = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     if len(faces) > 0:
#         x, y, w, h = faces[0]
#         pad = int(0.1 * min(w, h))
#         x1 = max(0, x - pad);              y1 = max(0, y - pad)
#         x2 = min(frame_bgr.shape[1], x+w+pad)
#         y2 = min(frame_bgr.shape[0], y+h+pad)
#         return frame_bgr[y1:y2, x1:x2]
#     return frame_bgr

# def _extract_vid_frames(video_path, sequence_length):
#     cap          = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
#     n            = min(sequence_length, total_frames)
#     indices      = [int(i * total_frames / n) for i in range(n)]
#     tensors, display_frames = [], []
#     for idx in indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
#         if not ret: continue
#         face = _detect_and_crop_face(frame)
#         rgb  = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#         pil  = PILImage.fromarray(rgb).resize((VID_IM_SIZE, VID_IM_SIZE), PILImage.LANCZOS)
#         # FIX: use _pil_to_tensor() instead of torchvision VID_TRANSFORM to
#         # avoid mode_to_nptype crash in torchvision 0.16 ToTensor()
#         tensors.append(_pil_to_tensor(pil))
#         display_frames.append((idx / fps, pil))
#     cap.release()
#     if not tensors: return None, []
#     while len(tensors) < sequence_length: tensors.append(tensors[-1])
#     stacked = torch.stack(tensors[:sequence_length])
#     return stacked.unsqueeze(0), display_frames

# def _get_vid_info(video_path):
#     cap     = cv2.VideoCapture(video_path)
#     fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
#     total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()
#     dur     = total / fps
#     dur_str = (f"{int(dur//60)}m {int(dur%60)}s" if dur >= 60 else f"{dur:.1f}s")
#     return {"fps": fps, "total": total, "w": w, "h": h, "dur": dur, "dur_str": dur_str}

# def _run_vid_prediction(model, device, video_tensor):
#     video_tensor = video_tensor.float().to(device)
#     with torch.no_grad():
#         fmap, logits = model(video_tensor)
#         probs = VF.softmax(logits, dim=1)[0]
#     return int(probs.argmax().item()), probs[0].item(), probs[1].item()

# # ── UI ────────────────────────────────────────────────────────────────────────
# st.markdown('<div class="hero-title">🎬 Video Detector</div>', unsafe_allow_html=True)
# st.caption("ResNeXt50 + LSTM · face-aware · sequence modeling · 5 pretrained models · ensemble mode.")
# st.divider()

# vl, vr = st.columns([1, 1], gap="large")
# with vl:
#     uvid = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv","webm"], key="vup")
#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown('<div class="section-label-vid">Model</div>', unsafe_allow_html=True)
#     vid_model_choice = st.selectbox(
#         "model", list(VID_MODELS_AVAILABLE.keys()), index=0,
#         label_visibility="collapsed",
#         help="Higher frame count = more temporal context = slower but more accurate.",
#         key="vid_sel",
#     )
#     vid_model_filename, vid_seq_len = VID_MODELS_AVAILABLE[vid_model_choice]
#     st.caption(f"Sequence {vid_seq_len} frames · `{vid_model_filename}`")
#     st.markdown("<br>", unsafe_allow_html=True)
#     vid_run_all     = st.checkbox("Run all 5 models (ensemble)", value=False,
#                                   help="Runs every model and shows individual + majority-vote verdict.")
#     st.caption("Ensemble uses majority vote across all 5 models.")
#     st.markdown("<br>", unsafe_allow_html=True)
#     vid_analyze_btn = st.button("🎬 Analyze Video", type="primary", use_container_width=True, key="vbtn")
# with vr:
#     vid_result_ph = st.empty()

# if uvid and not vid_analyze_btn:
#     with vr: vid_result_ph.video(uvid)

# if uvid and vid_analyze_btn:
#     vid_bytes = uvid.read()
#     if len(vid_bytes) / (1024**2) > 500:
#         st.error("File too large. Max 500 MB."); st.stop()
#     suffix = os.path.splitext(uvid.name)[-1]
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp.write(vid_bytes); tmp_path = tmp.name
#     try:
#         info = _get_vid_info(tmp_path)
#         models_to_run = (list(VID_MODELS_AVAILABLE.items()) if vid_run_all
#                          else [(vid_model_choice, (vid_model_filename, vid_seq_len))])
#         all_results = []; display_frames = []
#         n_models = len(models_to_run)
#         progress = st.progress(0); status = st.empty()

#         for m_idx, (m_name, (m_file, m_seq)) in enumerate(models_to_run):
#             status.info(f"[{m_idx+1}/{n_models}] Loading {m_name}…")
#             try:    model_v, device_v = _load_vid_model(m_file)
#             except FileNotFoundError as e: st.error(str(e)); st.stop()
#             status.info(f"[{m_idx+1}/{n_models}] Extracting {m_seq} frames + face detection…")
#             video_tensor, display_frames = _extract_vid_frames(tmp_path, m_seq)
#             if video_tensor is None:
#                 st.error("Could not extract frames. Try a different video format."); st.stop()
#             status.info(f"[{m_idx+1}/{n_models}] Running inference…")
#             pred, fake_p, real_p = _run_vid_prediction(model_v, device_v, video_tensor)
#             all_results.append((m_name, pred, fake_p, real_p))
#             progress.progress((m_idx + 1) / n_models)

#         progress.empty(); status.empty()

#         fake_votes = sum(1 for _, p, _, _ in all_results if p == 0)
#         real_votes = len(all_results) - fake_votes
#         is_fake    = fake_votes >= real_votes if vid_run_all else (all_results[0][1] == 0)
#         avg_fake   = float(np.mean([fp for _,_,fp,_ in all_results]))
#         avg_real   = float(np.mean([rp for _,_,_,rp in all_results]))
#         verdict    = "AI Generated" if is_fake else "Real"
#         css_cls    = "fake"      if is_fake else "real"
#         conf_pct   = int(avg_fake*100) if is_fake else int(avg_real*100)

#         with vr:
#             vid_result_ph.markdown(
#                 result_card_html(
#                     verdict, css_cls, conf_pct,
#                     avg_fake if is_fake else avg_real,
#                     (f'<b>{"Ensemble verdict" if vid_run_all else "Verdict"}</b>'
#                      f'{"&nbsp;·&nbsp; Fake " + str(fake_votes) + "/" + str(len(all_results)) + " models" if vid_run_all else ""}'
#                      f'<br><b>Fake score</b> · {avg_fake:.4f} &nbsp; <b>Real score</b> · {avg_real:.4f}<br>'
#                      f'<b>Resolution</b> · {info["w"]}×{info["h"]} &nbsp;·&nbsp; <b>Duration</b> · {info["dur_str"]}<br>'
#                      f'<b>FPS</b> · {info["fps"]:.1f} &nbsp;·&nbsp; <b>Total frames</b> · {info["total"]:,}<br>'
#                      f'<b>Device</b> · {str(device_v).upper()} &nbsp;·&nbsp; <b>Model</b> · ResNeXt50+LSTM')
#                 ), unsafe_allow_html=True)

#         if vid_run_all:
#             st.markdown('<div class="section-label-vid" style="margin-top:1.2rem;">Per-model breakdown</div>', unsafe_allow_html=True)
#             cards_html = '<div class="model-cards">'
#             for m_name, pred, fp, rp in all_results:
#                 mc_cls  = "fake" if pred == 0 else "real"
#                 mc_lbl  = "FAKE" if pred == 0 else "REAL"
#                 mc_conf = int(fp*100) if pred == 0 else int(rp*100)
#                 short   = m_name.split("·")[0].strip()
#                 cards_html += f'<div class="model-card"><div class="mc-name">{short}</div><div class="mc-score {mc_cls}">{mc_lbl}</div><div class="mc-conf">{mc_conf}% confidence</div></div>'
#             cards_html += '</div>'
#             st.markdown(cards_html, unsafe_allow_html=True)

#         n_f  = len(display_frames)
#         segs = "".join(
#             f'<div class="tl-seg tl-{"fake" if (i/max(n_f,1))<avg_fake else "real"}" title="frame {i}"></div>'
#             for i in range(n_f))
#         st.caption(f"Sequence timeline · {n_f} frames  🔴 Fake-leaning · 🟢 Real-leaning")
#         st.markdown(f'<div class="tl-bar">{segs}</div>', unsafe_allow_html=True)

#         st.caption("Sampled & face-cropped frames")
#         CPR = 8
#         for row in range(math.ceil(n_f / CPR)):
#             cols = st.columns(CPR)
#             for ci, col in enumerate(cols):
#                 idx2 = row * CPR + ci
#                 if idx2 >= n_f: break
#                 ts, pil = display_frames[idx2]
#                 with col:
#                     st.image(pil, use_container_width=True)
#                     st.markdown(f'<div class="frame-label {css_cls}">{verdict}<br><span style="opacity:0.5">{ts:.1f}s</span></div>', unsafe_allow_html=True)
#     finally:
#         if os.path.exists(tmp_path): os.unlink(tmp_path)

# st.markdown("""
# <div class="info-box">
#     <b>Architecture:</b> ResNeXt50_32x4d → AdaptiveAvgPool2d(1) → LSTM(2048) → mean pool → Dropout(0.4) → Linear(2048,2).
#     Softmax: index 0 = FAKE · index 1 = REAL.<br><br>
#     <b>Preprocessing:</b> Evenly sampled frames · Haar cascade face crop (fallback: full frame) · 112×112 · ImageNet normalisation.<br><br>
#     <b>Models:</b> FaceForensics++ and Celeb-DF · 40/60/80/100 frame sequences · 97% best accuracy.
# </div>""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
#  pages/3_🎬_Video_Detection.py
#  Deepfake Video Detector
#  Phase 1: C2PA manifest + SynthID text-marker provenance check (structural,
#           no ML — parses ISOBMFF boxes directly from the video file).
#  Phase 2: ResNeXt50 + LSTM visual classifier ("Video Detection Model"),
#           only runs if Phase 1 finds no provenance evidence.
# ─────────────────────────────────────────────────────────────────────────────
import os
import math
import struct
import tempfile
import numpy as np
import streamlit as st
from PIL import Image as PILImage

st.set_page_config(page_title="Video Detection · DeepSentinel", page_icon="🎬", layout="wide")

from utils import inject_css, result_card_html, ensure_file
inject_css()

import torch
import torch.nn as nn
import torch.nn.functional as VF
from torchvision import models as tv_models
import cv2

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ DeepSentinel")
    st.caption("AI & Deepfake Detection Suite")
    st.divider()
    st.caption("Model weights download automatically on first use and are cached for the session.")

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — VIDEO PROVENANCE CHECK (C2PA + SynthID text marker)
# ═════════════════════════════════════════════════════════════════════════════
#
#  Structural parse of the file's top-level ISOBMFF ("box"/"atom") structure,
#  looking for a 'uuid' box whose 16-byte identifier matches C2PA's registered
#  UUID (per the C2PA spec / AWS MediaConvert's documented embedding format).
#  This confirms a C2PA manifest box is present — it does NOT perform full
#  cryptographic signature validation (that requires the full C2PA SDK).
#
#  SynthID detection here is a weaker, best-effort text search for "SynthID"
#  mentions inside the video's own metadata atoms, in case a generation tool
#  also wrote a plain-text label. This is NOT real watermark verification —
#  a hit is a hint worth following up on, a miss just means "no label found."
# ─────────────────────────────────────────────────────────────────────────────

C2PA_UUID = bytes.fromhex("d8fec3d61b0e483c92975828877ec481")

SYNTHID_TEXT_MARKERS = [b"synthid", b"SynthID"]
KNOWN_AI_VIDEO_TOOLS = [
    b"veo", b"sora", b"runway", b"pika", b"luma", b"kling",
    b"stable video diffusion", b"gen-3", b"gen-2", b"haiper",
]


def _iter_top_level_boxes(f):
    """Yields (box_type: bytes, box_size: int, box_start_offset: int) for
    each top-level ISOBMFF box, without loading the whole file into memory."""
    while True:
        header = f.read(8)
        if len(header) < 8:
            return
        size, box_type = struct.unpack(">I4s", header)
        start = f.tell() - 8

        if size == 1:
            ext = f.read(8)
            if len(ext) < 8:
                return
            size = struct.unpack(">Q", ext)[0]
            header_len = 16
        elif size == 0:
            remaining = f.seek(0, 2) - start
            f.seek(start + 8)
            size = remaining
            header_len = 8
        else:
            header_len = 8

        yield box_type, size, start

        next_pos = start + size
        if size < header_len or next_pos <= start:
            return  # malformed box, stop rather than loop forever
        f.seek(next_pos)


def check_c2pa_video(video_path: str) -> dict:
    """Scans top-level ISOBMFF boxes for a 'uuid' box matching C2PA's
    registered identifier."""
    hits = []
    try:
        with open(video_path, "rb") as f:
            is_isobmff = False
            for box_type, size, start in _iter_top_level_boxes(f):
                if box_type == b"ftyp":
                    is_isobmff = True

                if box_type == b"uuid":
                    f.seek(start + 8)
                    uuid_bytes = f.read(16)
                    if uuid_bytes == C2PA_UUID:
                        f.seek(start + 8 + 16 + 4)  # skip 4 reserved bytes
                        purpose = b""
                        for _ in range(32):
                            b = f.read(1)
                            if not b or b == b"\x00":
                                break
                            purpose += b
                        hits.append({
                            "tag": "C2PA", "tag_class": "tag-c2pa",
                            "label": "C2PA manifest box (ISOBMFF)",
                            "text": f"C2PA UUID box found, purpose={purpose.decode('utf-8', errors='replace') or 'unknown'}",
                        })
                    f.seek(start + size)

            if not is_isobmff:
                return {"applicable": False, "hits": [], "note": "Not a recognized ISOBMFF (MP4/MOV) container — skipped."}

    except (OSError, struct.error) as e:
        return {"applicable": False, "hits": [], "note": f"Could not parse file structure: {e}"}

    return {"applicable": True, "hits": hits}


def check_synthid_text_marker(video_path: str, scan_bytes: int = 20_000_000) -> dict:
    """Best-effort text search for 'SynthID' or known AI video-tool names.
    NOT real watermark detection — see module note above."""
    hits = []
    try:
        with open(video_path, "rb") as f:
            data = f.read(scan_bytes)
    except OSError as e:
        return {"hits": [], "note": f"Could not read file: {e}"}

    lower = data.lower()
    for marker in SYNTHID_TEXT_MARKERS:
        if marker.lower() in lower:
            hits.append({
                "tag": "SYNTHID", "tag_class": "tag-synthid",
                "label": "SynthID text mention",
                "text": "'SynthID' appears in the file's own bytes/metadata — text label hint, not a verified watermark check.",
            })
            break

    for tool in KNOWN_AI_VIDEO_TOOLS:
        if tool in lower:
            hits.append({
                "tag": "TOOL", "tag_class": "tag-tool",
                "label": "AI video tool mention",
                "text": f"Metadata mentions a known AI video generation tool: {tool.decode()}",
            })
            break

    return {"hits": hits}


def run_phase1_video(video_path: str) -> dict:
    """Combined Phase 1 video provenance check. Returns a flat evidence list —
    the 'tag' field distinguishes strong (C2PA) vs weak (text marker) signals."""
    c2pa_result = check_c2pa_video(video_path)
    synthid_result = check_synthid_text_marker(video_path)
    evidence = list(c2pa_result.get("hits", [])) + list(synthid_result.get("hits", []))
    return {
        "evidence": evidence,
        "container_recognized": c2pa_result.get("applicable", False),
        "note": c2pa_result.get("note"),
    }


# ── Phase 1 rendering ────────────────────────────────────────────────────────
st.markdown("""
<style>
.verdict-ai-vid {
    display: inline-flex; align-items: center; gap: 10px;
    background: #052e16; border: 1px solid #166534;
    border-radius: 999px; padding: 9px 20px; margin: 20px 0 0;
}
.verdict-ai-vid .check { color: #4ade80; font-size: 15px; }
.verdict-ai-vid .title { color: #4ade80; font-weight: 600; font-size: 14px; }
.verdict-clean-vid {
    display: inline-flex; align-items: center; gap: 10px;
    background: #141414; border: 1px solid #2a2a2a;
    border-radius: 999px; padding: 9px 20px; margin: 20px 0 0;
}
.verdict-clean-vid .title { color: #6b7280; font-weight: 600; font-size: 14px; }
.evidence-header-vid {
    font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
    color: #4b5563; text-transform: uppercase; margin: 18px 0 8px;
}
.ev-row-vid {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 0; border-bottom: 1px solid #1a1a1a;
}
.ev-row-vid:last-child { border-bottom: none; }
.ev-tag-vid {
    font-size: 9px; font-weight: 700; letter-spacing: 0.07em;
    padding: 3px 8px; border-radius: 4px; flex-shrink: 0; text-transform: uppercase;
}
.tag-c2pa    { background:#1c1007; color:#fdba74; border:1px solid #92400e; }
.tag-synthid { background:#042f2e; color:#5eead4; border:1px solid #134e4a; }
.tag-tool    { background:#0f172a; color:#93c5fd; border:1px solid #1e3a8a; }
.ev-text-vid {
    font-size: 12.5px; color: #9ca3af;
    background: #1a1a1a; border: 1px solid #272727;
    border-radius: 5px; padding: 2px 8px; font-family: 'Courier New', monospace;
}
.notmean-block-vid {
    background: #0d0d0d; border: 1px solid #1e1e1e; border-left: 3px solid #27272a;
    border-radius: 8px; padding: 18px 22px; margin: 16px 0 0;
    font-size: 13px; color: #4b5563; line-height: 1.75;
}
.notmean-block-vid strong {
    color: #6b7280; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.06em; display: block; margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


def render_phase1_evidence(evidence):
    rows = ""
    for ev in evidence:
        rows += (
            f'<div class="ev-row-vid">'
            f'<span class="ev-tag-vid {ev["tag_class"]}">{ev["tag"]}</span>'
            f'<span class="ev-text-vid">{ev["text"]}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div class="evidence-header-vid">Evidence found</div>'
        f'<div style="background:#0d0d0d;border:1px solid #1a1a1a;border-radius:10px;padding:4px 16px">{rows}</div>',
        unsafe_allow_html=True,
    )


def render_phase1_ai_verdict(evidence):
    has_c2pa = any(e["tag"] == "C2PA" for e in evidence)
    st.markdown(
        '<div class="verdict-ai-vid">'
        '<span class="check">✓</span>'
        f'<span class="title">{"AI provenance verified" if has_c2pa else "Provenance hint found"}</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    render_phase1_evidence(evidence)


def render_phase1_clean_verdict():
    st.markdown(
        '<div class="verdict-clean-vid"><span class="title">No provenance metadata</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notmean-block-vid">'
        '<strong>What this does NOT mean</strong>'
        '"No provenance metadata" does NOT mean the video is real / human-made. '
        'It means the file has no C2PA manifest box or SynthID text label detectable '
        'in its own bytes. Re-encoding, re-uploading to social platforms, and screen '
        'recording all strip this kind of metadata routinely — so a miss here is common '
        'even for genuinely AI-generated video. Running the visual classifier below.'
        '</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — VISUAL CLASSIFIER (single model)
# ═════════════════════════════════════════════════════════════════════════════

class _VideoModel(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1,
                 hidden_dim=2048, bidirectional=False):
        super().__init__()
        backbone   = tv_models.resnext50_32x4d(weights=None)
        self.model = nn.Sequential(*list(backbone.children())[:-2])
        self.lstm  = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu  = nn.LeakyReLU()
        self.dp    = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x    = self.avgpool(fmap)
        x    = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

VID_IM_SIZE = 112

_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def _pil_to_tensor(pil_img):
    """Version-safe PIL Image → normalised float32 tensor (3, H, W)."""
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(2, 0, 1)
    return (t - _NORM_MEAN) / _NORM_STD

# Single model only.
VID_MODEL_FILE  = "model_95_acc_40_frames_FF_data.pt"
VID_SEQ_LEN     = 40
VID_MODEL_LABEL = "Video Detection Model"

@st.cache_resource(show_spinner=False)
def _load_vid_model(model_filename):
    """Loads and caches the video model, keyed on model_filename — downloads
    and loads at most once per app lifetime."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = ensure_file(model_filename)   # downloads once, returns actual on-disk path
    m = _VideoModel(num_classes=2)
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.to(device).eval()
    return m, device

def _detect_and_crop_face(frame_bgr):
    gray     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces    = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        pad = int(0.1 * min(w, h))
        x1 = max(0, x - pad);              y1 = max(0, y - pad)
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
        pil  = PILImage.fromarray(rgb).resize((VID_IM_SIZE, VID_IM_SIZE), PILImage.LANCZOS)
        tensors.append(_pil_to_tensor(pil))
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

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎬 Video Detector</div>', unsafe_allow_html=True)
st.caption("Phase 1: C2PA/SynthID provenance scan · Phase 2: ResNeXt50 + LSTM visual classifier.")
st.divider()

vl, vr = st.columns([1, 1], gap="large")
with vl:
    uvid = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv","webm"], key="vup")
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"Model · {VID_MODEL_LABEL} ({VID_SEQ_LEN} frames · `{VID_MODEL_FILE}`)")
    st.markdown("<br>", unsafe_allow_html=True)
    vid_analyze_btn = st.button("🎬 Analyze Video", type="primary", use_container_width=True, key="vbtn")
with vr:
    vid_result_ph = st.empty()

if uvid and not vid_analyze_btn:
    with vr: vid_result_ph.video(uvid)

if uvid and vid_analyze_btn:
    vid_bytes = uvid.read()
    if len(vid_bytes) / (1024**2) > 500:
        st.error("File too large. Max 500 MB."); st.stop()
    suffix = os.path.splitext(uvid.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(vid_bytes); tmp_path = tmp.name
    try:
        # ── Phase 1: provenance check ────────────────────────────────────────
        with st.spinner("Scanning provenance metadata…"):
            p1 = run_phase1_video(tmp_path)

        if p1["evidence"]:
            render_phase1_ai_verdict(p1["evidence"])
            st.caption("Provenance evidence found — skipping the visual classifier.")
        else:
            render_phase1_clean_verdict()
            st.divider()

            # ── Phase 2: visual classifier ───────────────────────────────────
            info = _get_vid_info(tmp_path)
            status = st.empty()

            status.info("Loading model…")
            try:    model_v, device_v = _load_vid_model(VID_MODEL_FILE)
            except FileNotFoundError as e: st.error(str(e)); st.stop()

            status.info(f"Extracting {VID_SEQ_LEN} frames + face detection…")
            video_tensor, display_frames = _extract_vid_frames(tmp_path, VID_SEQ_LEN)
            if video_tensor is None:
                st.error("Could not extract frames. Try a different video format."); st.stop()

            status.info("Running inference…")
            pred, fake_p, real_p = _run_vid_prediction(model_v, device_v, video_tensor)
            status.empty()

            is_fake  = (pred == 0)
            verdict  = "AI Generated" if is_fake else "Real"
            css_cls  = "fake" if is_fake else "real"
            conf_pct = int(fake_p*100) if is_fake else int(real_p*100)

            with vr:
                vid_result_ph.markdown(
                    result_card_html(
                        verdict, css_cls, conf_pct,
                        fake_p if is_fake else real_p,
                        (f'<b>Verdict</b><br>'
                         f'<b>Fake score</b> · {fake_p:.4f} &nbsp; <b>Real score</b> · {real_p:.4f}<br>'
                         f'<b>Resolution</b> · {info["w"]}×{info["h"]} &nbsp;·&nbsp; <b>Duration</b> · {info["dur_str"]}<br>'
                         f'<b>FPS</b> · {info["fps"]:.1f} &nbsp;·&nbsp; <b>Total frames</b> · {info["total"]:,}<br>'
                         f'<b>Device</b> · {str(device_v).upper()} &nbsp;·&nbsp; <b>Model</b> · {VID_MODEL_LABEL}')
                    ), unsafe_allow_html=True)

            n_f  = len(display_frames)
            segs = "".join(
                f'<div class="tl-seg tl-{"fake" if (i/max(n_f,1))<fake_p else "real"}" title="frame {i}"></div>'
                for i in range(n_f))
            st.caption(f"Sequence timeline · {n_f} frames  🔴 Fake-leaning · 🟢 Real-leaning")
            st.markdown(f'<div class="tl-bar">{segs}</div>', unsafe_allow_html=True)

            st.caption("Sampled & face-cropped frames")
            CPR = 8
            for row in range(math.ceil(n_f / CPR)):
                cols = st.columns(CPR)
                for ci, col in enumerate(cols):
                    idx2 = row * CPR + ci
                    if idx2 >= n_f: break
                    ts, pil = display_frames[idx2]
                    with col:
                        st.image(pil, use_container_width=True)
                        st.markdown(f'<div class="frame-label {css_cls}">{verdict}<br><span style="opacity:0.5">{ts:.1f}s</span></div>', unsafe_allow_html=True)
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

st.markdown(f"""
<div class="info-box">
    <b>Phase 1 — Provenance:</b> Structural scan for a C2PA manifest box (ISOBMFF 'uuid' box,
    per the C2PA spec) plus a best-effort text search for SynthID / known AI video-tool mentions
    in the file's own metadata. Presence detection only — not full cryptographic signature
    validation, and a miss does not mean the video is human-made.<br><br>
    <b>Phase 2 — {VID_MODEL_LABEL}:</b> ResNeXt50_32x4d → AdaptiveAvgPool2d(1) → LSTM(2048) →
    mean pool → Dropout(0.4) → Linear(2048,2). Softmax: index 0 = FAKE · index 1 = REAL.
    {VID_SEQ_LEN} evenly sampled frames · Haar cascade face crop (fallback: full frame) ·
    112×112 · ImageNet normalisation. Trained on FaceForensics++ · 95% accuracy.
</div>""", unsafe_allow_html=True)
