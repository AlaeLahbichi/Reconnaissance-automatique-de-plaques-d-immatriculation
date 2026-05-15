import streamlit as st
import cv2
import os
import numpy as np
import tempfile
from ultralytics import YOLO
from PIL import Image
import easyocr
from paddleocr import PaddleOCR
import re
import time
import html


# --- CONFIGURATION ---
MODEL_PATH = "/Users/alaethelegend/Documents/Projet_DeepLearning/best.pt"
OUTPUT_FOLDERS = ["outputs/images", "outputs/crops", "outputs/videos"]
for folder in OUTPUT_FOLDERS:
    os.makedirs(folder, exist_ok=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;600&display=swap');

:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --card:      #161a22;
    --border:    #1f2535;
    --accent:    #00e5ff;
    --accent2:   #ff3c5f;
    --accent3:   #7c3aed;
    --text:      #e2e8f0;
    --muted:     #4a5568;
    --success:   #00ff9d;
    --warning:   #ffcc00;
    --radius:    12px;
}

/* ── Base ────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stSidebar"] { display: none; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 2rem 3rem !important; max-width: 1300px; }

/* ── Hero banner ─────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161a22 50%, #0d1117 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 80% at 80% 50%, rgba(0,229,255,.06), transparent);
}
.hero-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: .7rem;
    letter-spacing: .2em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: .5rem;
}
.hero h1 {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    letter-spacing: .05em;
    background: linear-gradient(90deg, #fff 30%, var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
    line-height: 1 !important;
}
.hero p {
    color: var(--muted);
    font-size: .95rem;
    margin-top: .7rem;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,229,255,.1);
    border: 1px solid rgba(0,229,255,.3);
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    font-size: .65rem;
    padding: .25rem .7rem;
    border-radius: 999px;
    letter-spacing: .1em;
    margin-top: 1.2rem;
}

/* ── Mode selector ───────────────────────────────────────────── */
[data-testid="stRadio"] > label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .7rem !important;
    letter-spacing: .15em;
    color: var(--muted) !important;
    text-transform: uppercase;
    margin-bottom: .5rem;
}
[data-testid="stRadio"] [role="radiogroup"] {
    gap: .75rem !important;
    display: flex !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: .75rem 2rem !important;
    transition: all .2s !important;
    cursor: pointer !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    border-color: var(--accent) !important;
    background: rgba(0,229,255,.06) !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] {
    border-color: var(--accent) !important;
    background: rgba(0,229,255,.1) !important;
    box-shadow: 0 0 20px rgba(0,229,255,.15) !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"] span:last-child {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: var(--text) !important;
}

/* ── File uploader ───────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 2rem !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .75rem !important;
    letter-spacing: .1em;
    text-transform: uppercase;
}
[data-testid="stFileUploadDropzone"] button {
    background: rgba(0,229,255,.1) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .75rem !important;
    border-radius: 8px !important;
}

/* ── Image containers ────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}

/* ── Success / Info / Warning boxes ─────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    border: none !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .8rem !important;
}
.stSuccess {
    background: rgba(0,255,157,.08) !important;
    border-left: 3px solid var(--success) !important;
    color: var(--success) !important;
}
.stInfo {
    background: rgba(0,229,255,.08) !important;
    border-left: 3px solid var(--accent) !important;
    color: var(--accent) !important;
}
.stWarning {
    background: rgba(255,204,0,.08) !important;
    border-left: 3px solid var(--warning) !important;
    color: var(--warning) !important;
}

/* ── Metric / Counter card ───────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem 1.5rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .65rem !important;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
}

/* ── Progress bar ────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--accent3), var(--accent)) !important;
    border-radius: 999px !important;
}
[data-testid="stProgressBar"] > div {
    background: var(--border) !important;
    border-radius: 999px !important;
    height: 6px !important;
}

/* ── Plate result card (custom HTML) ────────────────────────── */
.plate-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    margin-top: .75rem;
    position: relative;
}
.plate-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
    border-radius: var(--radius) var(--radius) 0 0;
}
.plate-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: .6rem;
    letter-spacing: .2em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: .4rem;
}
.plate-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: .12em;
    text-shadow: 0 0 20px rgba(0,229,255,.4);
}
.plate-empty {
    font-family: 'JetBrains Mono', monospace;
    font-size: .75rem;
    color: var(--muted);
}

/* ── Section divider ─────────────────────────────────────────── */
.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: .65rem;
    letter-spacing: .25em;
    color: var(--muted);
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: .8rem;
    margin: 1.5rem 0 1rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Columns gap ─────────────────────────────────────────────── */
[data-testid="column"] { padding: 0 .5rem !important; }

/* ── Video ───────────────────────────────────────────────────── */
video { border-radius: var(--radius) !important; width: 100% !important; }

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model = YOLO(MODEL_PATH)
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader 

model, reader = load_models()


def enhance_plate_v2(crop):
    h, w = crop.shape[:2]
    img = cv2.resize(crop, (w*4, h*4), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return processed

def perform_ocr(crop):
    processed_img = enhance_plate_v2(crop)
    my_whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ|-"
    results = reader.readtext(
        processed_img,
        detail=1,
        allowlist=my_whitelist,
        paragraph=False,
        mag_ratio=1.5,
        text_threshold=0.7,
        low_text=0.3
    )
    results = sorted(results, key=lambda x: x[0][0][0])
    final_parts = []
    last_x = None
    for res in results:
        pos, text, conf = res
        if conf > 0.2:
            current_x = pos[0][0]
            if last_x is not None and (current_x - last_x) > (processed_img.shape[1] * 0.08):
                final_parts.append(" ")
            final_parts.append(text)
            last_x = pos[1][0]
    raw_text = "".join(final_parts)
    cleaned = re.sub(r'[^a-zA-Z0-9\|\-\s]', '', raw_text)
    return cleaned.strip(), processed_img

st.markdown("""
<div class="hero">
    <div class="hero-label">Computer Vision · ALPR System</div>
    <h1>PLATE AI PRO</h1>
    <p>Détection & reconnaissance de plaques d'immatriculation</p>
    <span class="hero-badge">YOLOv8 & EasyOCR</span>
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="section-title">Mode d\'analyse</div>', unsafe_allow_html=True)
mode = st.radio("", ["🖼  Image", "🎥  Vidéo"], horizontal=True, label_visibility="collapsed")

st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)


if "Image" in mode:
    up_file = st.file_uploader("Glissez-déposez une photo ou cliquez pour parcourir",
                                type=["jpg", "jpeg", "png"],
                                label_visibility="visible")
    if up_file:
        img_np = np.array(Image.open(up_file))

        results = model(img_np, conf=0.55)
        boxes = results[0].boxes

        if boxes and len(boxes) > 0:
            st.markdown('<div class="section-title">Plaques détectées</div>', unsafe_allow_html=True)
            n_cols = min(len(boxes), 3)
            cols = st.columns(n_cols)

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img_np[y1:y2, x1:x2]
                text, processed_crop = perform_ocr(crop)

                # Annotation visuelle
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 229, 255), 3)
                cv2.putText(img_np, text, (x1, y1 - 12), 0, 1.1, (255, 60, 95), 3)

                with cols[i % n_cols]:
                    st.image(processed_crop, caption=f"Prétraitement · Plaque {i+1}", use_column_width=True)
                    plate_html = f"""
                    <div class="plate-card">
                        <div class="plate-label">// Plaque {i+1}</div>
                        <div class="plate-text">{text if text else '—'}</div>
                        {"" if text else '<div class="plate-empty">Aucun texte détecté</div>'}
                    </div>
                    """
                    st.markdown(plate_html, unsafe_allow_html=True)

            st.markdown('<div class="section-title">Vue annotée</div>', unsafe_allow_html=True)
            st.image(img_np, channels="RGB", use_column_width=True)

        else:
            st.markdown("""
            <div style="background:rgba(255,204,0,.07);border:1px solid rgba(255,204,0,.25);border-left:3px solid #ffcc00;
                        border-radius:12px;padding:1.2rem 1.5rem;font-family:'JetBrains Mono',monospace;
                        font-size:.8rem;color:#ffcc00;margin-top:1rem;">
                ⚠ &nbsp;Aucune plaque détectée dans cette image.
            </div>
            """, unsafe_allow_html=True)


elif "Vidéo" in mode:
    uploaded_video = st.file_uploader("Glissez-déposez une vidéo ou cliquez pour parcourir",
                                       type=["mp4", "avi", "mov"],
                                       label_visibility="visible")

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25

        filename    = uploaded_video.name
        name        = os.path.splitext(filename)[0]

        os.makedirs("outputs/videos", exist_ok=True)
        os.makedirs("outputs/plates", exist_ok=True)

        output_path = os.path.abspath(os.path.join("outputs/videos", f"{name}.mp4"))

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        st.markdown('<div class="section-title">Traitement en direct</div>', unsafe_allow_html=True)
        col_vid, col_stats = st.columns([3, 1])

        with col_vid:
            stframe = st.empty()

        with col_stats:
            st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
            counter_placeholder = st.empty()
            status_placeholder  = st.empty()

        st.markdown('<div class="section-title">Plaques capturées avec OCR</div>', unsafe_allow_html=True)
        ocr_gallery_placeholder = st.empty()

        progress = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current      = 0

        unique_ids   = set()
        ocr_done_ids = set()
        plate_results = {}

        def render_ocr_gallery():
            if not plate_results:
                ocr_gallery_placeholder.markdown("""
                <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.12);
                            border-radius:12px;padding:1rem 1.3rem;font-family:'JetBrains Mono',monospace;
                            font-size:.75rem;color:rgba(255,255,255,.65);">
                    Aucune plaque capturée pour le moment.
                </div>
                """, unsafe_allow_html=True)
                return

            with ocr_gallery_placeholder.container():
                items = list(plate_results.items())
                n_cols = min(len(items), 3)
                cols = st.columns(n_cols)

                for idx, (plate_id, data) in enumerate(items):
                    with cols[idx % n_cols]:
                        st.image(
                            data["processed_crop"],
                            caption=f"Prétraitement · ID {plate_id}",
                            use_column_width=True
                        )

                        safe_text = html.escape(data["text"]) if data["text"] else "—"

                        plate_html = f"""
                        <div class="plate-card">
                            <div class="plate-label">// Plaque ID {plate_id}</div>
                            <div class="plate-text">{safe_text}</div>
                            {"" if data["text"] else '<div class="plate-empty">Aucun texte détecté</div>'}
                        </div>
                        """

                        st.markdown(plate_html, unsafe_allow_html=True)

        status_placeholder.markdown("""
        <div style="background:rgba(0,229,255,.07);border:1px solid rgba(0,229,255,.2);
                    border-radius:10px;padding:.9rem 1rem;font-family:'JetBrains Mono',monospace;
                    font-size:.7rem;color:#00e5ff;letter-spacing:.1em;">
            ⬤ &nbsp;ANALYSE EN COURS…
        </div>
        """, unsafe_allow_html=True)

        render_ocr_gallery()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True)
            boxes   = results[0].boxes

            if boxes is not None and boxes.id is not None:
                ids = boxes.id.cpu().numpy()

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    obj_id = int(ids[i])

                    unique_ids.add(obj_id)

                    # Sécuriser les coordonnées
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    # OCR une seule fois par ID
                    if obj_id not in ocr_done_ids:
                        crop_bgr = frame[y1:y2, x1:x2]

                        if crop_bgr.size > 0:
                            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

                            text, processed_crop = perform_ocr(crop_rgb)

                            snapshot_path = os.path.abspath(
                                os.path.join("outputs/plates", f"{name}_plate_ID_{obj_id}.jpg")
                            )

                            cv2.imwrite(snapshot_path, crop_bgr)

                            plate_results[obj_id] = {
                                "text": text,
                                "processed_crop": processed_crop,
                                "snapshot_path": snapshot_path
                            }

                            ocr_done_ids.add(obj_id)
                            render_ocr_gallery()

                    plate_text = plate_results.get(obj_id, {}).get("text", "")
                    label = f"ID {obj_id}"

                    if plate_text:
                        label += f" | {plate_text}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 229, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 60, 95), 2)

            total_plates = len(unique_ids)

            cv2.putText(frame, f"PLATES: {total_plates}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 229, 255), 2)

            out.write(frame)
            stframe.image(frame, channels="BGR", width="stretch")

            counter_placeholder.metric("Plaques uniques", total_plates)

            current += 1
            if total_frames > 0:
                progress.progress(min(current / total_frames, 1.0))

        cap.release()
        out.release()
        time.sleep(1)

        status_placeholder.markdown("""
        <div style="background:rgba(0,255,157,.07);border:1px solid rgba(0,255,157,.25);
                    border-radius:10px;padding:.9rem 1rem;font-family:'JetBrains Mono',monospace;
                    font-size:.7rem;color:#00ff9d;letter-spacing:.1em;">
            ✔ &nbsp;TRAITEMENT TERMINÉ
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Vidéo traitée</div>', unsafe_allow_html=True)
        st.video(output_path)

        st.markdown(f"""
        <div style="background:rgba(0,255,157,.06);border:1px solid rgba(0,255,157,.2);border-left:3px solid #00ff9d;
                    border-radius:12px;padding:1rem 1.4rem;font-family:'JetBrains Mono',monospace;
                    font-size:.75rem;color:#00ff9d;margin-top:.75rem;">
            ✔ &nbsp;Fichier sauvegardé → <span style="opacity:.6">{output_path}</span>
        </div>
        """, unsafe_allow_html=True)
