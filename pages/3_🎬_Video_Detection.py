# ─────────────────────────────────────────────────────────────────────────────
#  pages/3_🎬_Video_Detection.py
#  Deepfake Video Detector — ResNeXt50 + LSTM · face-aware · ensemble
#  Only torch/torchvision/cv2 loads when this page is visited.
# ─────────────────────────────────────────────────────────────────────────────
import os
import math
import tempfile
import numpy as np
import streamlit as st
from PIL import Image as PILImage

st.set_page_config(page_title="Video Detection · DeepSentinel", page_icon="🎬", layout="wide")

from utils import inject_css, result_card_html
inject_css()

import torch
import torch.nn as nn
import torch.nn.functional as VF
from torchvision import models as tv_models, transforms as tv_transforms
import cv2

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ DeepSentinel")
    st.caption("AI & Deepfake Detection Suite")
    st.divider()
    st.markdown("### 📁 Model Directory")
    vid_model_dir = st.text_input("Model directory", value=os.environ.get("VID_MODEL_DIR", "."),
                                  help="Directory where .pt files are saved")

# ── Model definition ──────────────────────────────────────────────────────────
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

VID_IM_SIZE   = 112
VID_TRANSFORM = tv_transforms.Compose([
    tv_transforms.ToPILImage(),
    tv_transforms.Resize((VID_IM_SIZE, VID_IM_SIZE)),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VID_MODELS_AVAILABLE = {
    "97% acc · 100 frames · FF"       : ("model_97_acc_100_frames_FF_data.pt",        100),
    "97% acc · 80 frames · FF"        : ("model_97_acc_80_frames_FF_data.pt",           80),
    "97% acc · 60 frames · FF"        : ("model_97_acc_60_frames_FF_data.pt",           60),
    "95% acc · 40 frames · FF"        : ("model_95_acc_40_frames_FF_data.pt",           40),
    "93% acc · 100 frames · Celeb+FF" : ("model_93_acc_100_frames_celeb_FF_data.pt",  100),
}

# ── Model loaders & helpers ───────────────────────────────────────────────────
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

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎬 Video Detector</div>', unsafe_allow_html=True)
st.caption("ResNeXt50 + LSTM · face-aware · sequence modeling · 5 pretrained models · ensemble mode.")
st.divider()

vl, vr = st.columns([1, 1], gap="large")
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
    vid_run_all     = st.checkbox("Run all 5 models (ensemble)", value=False,
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
    if len(vid_bytes) / (1024**2) > 500:
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
            progress.progress((m_idx + 1) / n_models)

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
                mc_cls  = "fake" if pred == 0 else "real"
                mc_lbl  = "FAKE" if pred == 0 else "REAL"
                mc_conf = int(fp*100) if pred == 0 else int(rp*100)
                short   = m_name.split("·")[0].strip()
                cards_html += f'<div class="model-card"><div class="mc-name">{short}</div><div class="mc-score {mc_cls}">{mc_lbl}</div><div class="mc-conf">{mc_conf}% confidence</div></div>'
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

        n_f  = len(display_frames)
        segs = "".join(
            f'<div class="tl-seg tl-{"fake" if (i/max(n_f,1))<avg_fake else "real"}" title="frame {i}"></div>'
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

st.markdown("""
<div class="info-box">
    <b>Architecture:</b> ResNeXt50_32x4d → AdaptiveAvgPool2d(1) → LSTM(2048) → mean pool → Dropout(0.4) → Linear(2048,2).
    Softmax: index 0 = FAKE · index 1 = REAL.<br><br>
    <b>Preprocessing:</b> Evenly sampled frames · Haar cascade face crop (fallback: full frame) · 112×112 · ImageNet normalisation.<br><br>
    <b>Models:</b> FaceForensics++ and Celeb-DF · 40/60/80/100 frame sequences · 97% best accuracy.
</div>""", unsafe_allow_html=True)
