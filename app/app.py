"""
CéphaloAI — app.py
Insaf Saouiki — EMSI — Stage Cabinet Dr. Redouane Chiguer

Interface Streamlit professionnelle pour l'analyse céphalométrique.
"""

import streamlit as st
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io
import sys
import os
import json
import base64
from datetime import datetime
from pathlib import Path


# ── Chemins ───────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from geometry import (
    compute_steiner_analysis, format_results,
    extract_coords, LANDMARK_NAMES, STEINER_NORMS
)
from interpretation import generer_diagnostic_complet, diagnostic_to_dict

# ══════════════════════════════════════════════════════════════
# CONFIGURATION PAGE
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title   = 'CéphaloAI — Cabinet Dr. Chiguer',
    page_icon    = '🦷',
    layout       = 'wide',
    initial_sidebar_state = 'expanded'
)

# ══════════════════════════════════════════════════════════════
# CSS PERSONNALISÉ — Design médical professionnel
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset et base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg-primary   : #0A0F1E;
    --bg-secondary : #0F1629;
    --bg-card      : #141B2D;
    --bg-hover     : #1A2238;
    --accent-blue  : #3B82F6;
    --accent-teal  : #06B6D4;
    --accent-green : #10B981;
    --accent-amber : #F59E0B;
    --accent-red   : #EF4444;
    --text-primary : #F1F5F9;
    --text-secondary: #94A3B8;
    --text-muted   : #475569;
    --border       : #1E293B;
    --border-light : #243047;
    --shadow       : 0 4px 24px rgba(0,0,0,0.4);
    --radius       : 12px;
    --radius-sm    : 8px;
}

/* ── Body global ── */
html, body, .stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }

/* ── Header principal ── */
.cepha-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem 2rem;
    background: linear-gradient(135deg, #0F1629 0%, #141B2D 100%);
    border: 1px solid var(--border-light);
    border-radius: var(--radius);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.cepha-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(135deg,
        rgba(59,130,246,0.08) 0%,
        rgba(6,182,212,0.05) 100%);
    pointer-events: none;
}
.cepha-logo {
    font-size: 2.8rem;
    line-height: 1;
}
.cepha-title {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #3B82F6, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.cepha-subtitle {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin-top: 0.2rem;
}
.cepha-badge {
    margin-left: auto;
    padding: 0.4rem 0.9rem;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 20px;
    font-size: 0.75rem;
    color: var(--accent-blue);
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover {
    border-color: var(--border-light);
    transform: translateY(-1px);
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
    line-height: 1.1;
}
.metric-sub {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 0.3rem;
}

/* ── Section title ── */
.section-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    font-weight: 700;
    margin: 1.5rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Tableau de mesures ── */
.measure-row {
    display: flex;
    align-items: center;
    padding: 0.65rem 1rem;
    border-radius: var(--radius-sm);
    margin-bottom: 0.3rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    font-size: 0.875rem;
    transition: background 0.15s;
}
.measure-row:hover { background: var(--bg-hover); }
.measure-name {
    width: 160px;
    font-weight: 500;
    color: var(--text-primary);
}
.measure-val {
    width: 80px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.95rem;
}
.measure-norm {
    flex: 1;
    color: var(--text-muted);
    font-size: 0.8rem;
}
.measure-status {
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    min-width: 70px;
    text-align: center;
}
.status-NORMAL { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.2); }
.status-LEGER  { background: rgba(245,158,11,0.12); color: #F59E0B; border: 1px solid rgba(245,158,11,0.2); }
.status-MODERE { background: rgba(249,115,22,0.12); color: #F97316; border: 1px solid rgba(249,115,22,0.2); }
.status-SEVERE { background: rgba(239,68,68,0.12);  color: #EF4444; border: 1px solid rgba(239,68,68,0.2); }

/* ── Classe squelettique badge ── */
.classe-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.2rem;
    border-radius: var(--radius);
    font-weight: 700;
    font-size: 0.95rem;
    margin: 0.3rem;
}
.classe-I   { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); color: #10B981; }
.classe-II  { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.3); color: #F59E0B; }
.classe-III { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.3);  color: #EF4444; }

/* ── Diagnostic item ── */
.diag-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-blue);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.5rem;
}
.diag-item.severe { border-left-color: #EF4444; }
.diag-item.modere { border-left-color: #F97316; }
.diag-item.leger  { border-left-color: #F59E0B; }
.diag-item.normal { border-left-color: #10B981; }
.diag-text { font-size: 0.875rem; color: var(--text-primary); font-weight: 500; }
.diag-conseil { font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.3rem; }

/* ── Boutons ── */
.stButton > button {
    background: linear-gradient(135deg, #3B82F6, #06B6D4) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1.5rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s, transform 0.1s !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-light) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-blue) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #3B82F6, #06B6D4) !important;
    border-radius: 4px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.75rem 1.25rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-blue) !important;
    border-bottom-color: var(--accent-blue) !important;
    background: transparent !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

/* ── Warning / Info / Success ── */
.stAlert { border-radius: var(--radius-sm) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent-blue) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb {
    background: var(--border-light);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Hide Streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Charge le modèle heatmap entraîné (EfficientNet-B4 + décodeur)."""
    try:
        from heatmap_model import HeatmapCephaloNet
        model = HeatmapCephaloNet(pretrained=False)
        model_path = ROOT / 'models' / 'checkpoints' / 'best_model_heatmap.pth'
        if model_path.exists():
            checkpoint = torch.load(
                str(model_path), map_location='cpu', weights_only=False
            )
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            return model, True
        return model, False
    except Exception as e:
        return None, False


def preprocess_for_inference(img_array, img_size=512):
    """Prétraite une image pour l'inférence."""
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_gray = clahe.apply(img_gray)
    img_res  = cv2.resize(img_gray, (img_size, img_size))
    img_rgb  = cv2.cvtColor(img_res, cv2.COLOR_GRAY2RGB)
    img_f    = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_f = (img_f - mean) / std
    tensor = torch.from_numpy(img_f.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def predict_landmarks(model, img_array):
    """Prédit les 19 landmarks sur une image (via heatmaps)."""
    from heatmap_model import heatmaps_to_coords
    h, w = img_array.shape[:2]
    tensor = preprocess_for_inference(img_array)
    with torch.no_grad():
        heatmaps = model(tensor)          # [1, 19, 128, 128]
    coords_norm = heatmaps_to_coords(heatmaps.squeeze(0))   # (19, 2) normalisé [0,1]
    coords = coords_norm.copy()
    coords[:, 0] *= w
    coords[:, 1] *= h
    return coords


def draw_landmarks_on_image(img_array, landmarks_px):
    """
    Dessine les 19 landmarks et les lignes de Steiner sur la radio.
    Retourne l'image annotée.
    """
    img = img_array.copy()
    h, w = img.shape[:2]

    COLORS = {
        'median'    : (255, 107, 107),   # rouge — points médians
        'bilateral' : (78,  205, 196),   # teal — points bilatéraux
        'incisor'   : (255, 230, 109),   # jaune — incisives
    }

    GROUP = {
    'S':'median','Na':'median','A':'median','B':'median',
    'Pog':'median','Gn':'median','Me':'median',
    'ANS':'median','PNS':'median','Sn':'median','Pog_s':'median',
    'Go':'bilateral','Po':'bilateral','Or':'bilateral','Ar':'bilateral',
    'U1':'incisor','L1':'incisor','Ls':'incisor','Li':'incisor',
    }

    coords = {name: (int(landmarks_px[i,0]), int(landmarks_px[i,1]))
              for i, name in enumerate(LANDMARK_NAMES)}

    # Lignes de Steiner
    LINES = [
        ('S','Na', (100,149,237,180)),
        ('Na','A', (152,251,152,180)),
        ('Na','B', (255,165,0,180)),
        ('Go','Gn',(255,105,180,180)),
        ('ANS','PNS',(173,216,230,180)),
        ('U1','L1',(255,255,100,180)),
    ]

    overlay = img.copy()
    for p1_name, p2_name, color in LINES:
        if p1_name in coords and p2_name in coords:
            cv2.line(overlay, coords[p1_name], coords[p2_name],
                     color[:3], max(1, w//400), cv2.LINE_AA)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    # Points landmarks
    for name, (x, y) in coords.items():
        group = GROUP.get(name, 'median')
        color = COLORS[group]
        r     = max(4, w//200)
        cv2.circle(img, (x, y), r+2, (0, 0, 0), -1)
        cv2.circle(img, (x, y), r,   color,      -1)

        # Labels
        font_scale = max(0.3, w/3000)
        cv2.putText(img, name, (x+r+2, y+4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, name, (x+r+2, y+4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, 1, cv2.LINE_AA)

    return img


def render_measure_row(f):
    """Rendu HTML d'une ligne de mesure."""
    return f"""
    <div class="measure-row">
        <span class="measure-name">{f['label']}</span>
        <span class="measure-val" style="color:{'#10B981' if f['statut']=='NORMAL'
            else '#F59E0B' if f['statut']=='LEGER'
            else '#F97316' if f['statut']=='MODERE' else '#EF4444'}">
            {f['valeur']}{f['unite']}
        </span>
        <span class="measure-norm">Norme : {f['norme']}</span>
        <span class="measure-status status-{f['statut']}">{f['statut']}</span>
    </div>"""


def render_diag_item(item):
    """Rendu HTML d'un item de diagnostic."""
    severity_class = item['statut'].lower() if item['statut'] in ['SEVERE','MODERE','LEGER','NORMAL'] else 'normal'
    return f"""
    <div class="diag-item {severity_class}">
        <div class="diag-text">{item['diagnostic']}</div>
        <div class="diag-conseil">→ {item['conseil']}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════
# HISTORIQUE PATIENT — sauvegarde et comparaison de visites
# ══════════════════════════════════════════════════════════════

HISTORY_DIR = ROOT / 'data' / 'history'


def slugify(name):
    """Transforme un nom de patient en identifiant de dossier sûr."""
    s = ''.join(c if c.isalnum() else '_' for c in (name or 'patient').strip().lower())
    return s or 'patient'


def save_visit(patient_name, age, img_annotated, landmarks_px, results):
    """Enregistre une visite (image + landmarks + mesures) sur le disque."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(patient_name)
    patient_dir = HISTORY_DIR / slug
    patient_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_pil = Image.fromarray(img_annotated)
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    landmarks_list = landmarks_px.tolist() if hasattr(landmarks_px, 'tolist') else landmarks_px

    visit = {
        'date'        : datetime.now().strftime('%d/%m/%Y %H:%M'),
        'timestamp'   : timestamp,
        'patient_name': patient_name or 'Patient',
        'age'         : age or '—',
        'landmarks_px': landmarks_list,
        'results'     : results,
        'image_b64'   : img_b64,
    }
    with open(patient_dir / f'{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(visit, f)
    return patient_dir / f'{timestamp}.json'


def list_patients():
    """Liste les identifiants (slugs) de patients ayant au moins une visite."""
    if not HISTORY_DIR.exists():
        return []
    return sorted([d.name for d in HISTORY_DIR.iterdir() if d.is_dir()])


def list_visits(slug):
    """Liste toutes les visites d'un patient, triées par date."""
    patient_dir = HISTORY_DIR / slug
    if not patient_dir.exists():
        return []
    visits = []
    for f in sorted(patient_dir.glob('*.json')):
        with open(f, encoding='utf-8') as fh:
            visits.append(json.load(fh))
    return sorted(visits, key=lambda v: v['timestamp'])


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 1.5rem 0;">
        <div style="font-size:2.5rem;">🦷</div>
        <div style="font-weight:700; font-size:1.1rem; color:#F1F5F9;">CéphaloAI</div>
        <div style="font-size:0.75rem; color:#64748B; margin-top:0.2rem;">
            Cabinet Dr. Redouane Chiguer
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Patient</div>', unsafe_allow_html=True)
    nom_patient = st.text_input('Nom du patient', placeholder='Ex: Mohamed Alami',
                                label_visibility='collapsed')
    age_patient = st.text_input('Âge', placeholder='Ex: 14 ans',
                                label_visibility='collapsed')

    st.markdown('<div class="section-title">Radiographie</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        'Téléradiographie latérale',
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        label_visibility='collapsed',
        help='Formats acceptés : PNG, JPG, BMP, TIFF'
    )

    st.markdown('<div class="section-title">Paramètres</div>', unsafe_allow_html=True)
    resolution = st.selectbox(
        'Résolution radio',
        ['0.1 mm/pixel (standard)', '0.12 mm/pixel', '0.08 mm/pixel'],
        label_visibility='collapsed'
    )
    scale = float(resolution.split()[0])

    st.markdown("""
    <div style="margin-top:2rem; padding:1rem; background:#0F1629;
                border-radius:8px; border:1px solid #1E293B;">
        <div style="font-size:0.7rem; color:#475569; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">
            À propos
        </div>
        <div style="font-size:0.75rem; color:#64748B; line-height:1.5;">
            Système d'aide au diagnostic céphalométrique basé sur l'IA.
            Ce rapport ne remplace pas l'expertise clinique.
        </div>
        <div style="margin-top:0.75rem; font-size:0.7rem; color:#334155;">
            EfficientNet-B4 · Analyse de Steiner 1952
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ══════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="cepha-header">
    <div class="cepha-logo">🦷</div>
    <div>
        <div class="cepha-title">CéphaloAI</div>
        <div class="cepha-subtitle">
            Analyse Céphalométrique de Steiner — Cabinet Dr. Redouane Chiguer
        </div>
    </div>
    <div class="cepha-badge">IA · Orthodontie</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ÉTAT INITIAL — pas de radio uploadée
# ══════════════════════════════════════════════════════════════

if uploaded_file is None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Points détectés</div>
            <div class="metric-value" style="color:#3B82F6;">19</div>
            <div class="metric-sub">Landmarks céphalométriques</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Mesures calculées</div>
            <div class="metric-value" style="color:#06B6D4;">15</div>
            <div class="metric-sub">Angles et distances Steiner</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Modèle</div>
            <div class="metric-value" style="color:#10B981; font-size:1.1rem;">EfficientNet-B4</div>
            <div class="metric-sub">Transfer Learning · Wing Loss</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.info('📂 Chargez une téléradiographie latérale dans la barre latérale pour commencer l\'analyse.')
    st.stop()


# ══════════════════════════════════════════════════════════════
# TRAITEMENT DE L'IMAGE
# ══════════════════════════════════════════════════════════════

img_pil   = Image.open(uploaded_file).convert('RGB')
img_array = np.array(img_pil)
img_h, img_w = img_array.shape[:2]

# Charger le modèle
model, model_loaded = load_model()

current_file_id = uploaded_file.file_id
if st.session_state.get('current_file_id') != current_file_id:
    for key in ['landmarks_px', 'img_annotated', 'results', 'formatted',
                'diagnostic', 'selected_landmark']:
        st.session_state.pop(key, None)
    st.session_state['current_file_id'] = current_file_id

# Onglets principaux
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '🖼️  Radiographie',
    '📐 Mesures de Steiner',
    '🩺 Diagnostic',
    '📋 Rapport',
    '📈 Suivi'
])


# ══════════════════════════════════════════════════════════════
# ONGLET 1 — RADIOGRAPHIE + LANDMARKS
# ══════════════════════════════════════════════════════════════

with tab1:
    col_img, col_info = st.columns([2, 1])

    with col_img:
        if model_loaded and model is not None:
            if 'landmarks_px' not in st.session_state:
                with st.spinner('Détection des landmarks en cours...'):
                    landmarks_px = predict_landmarks(model, img_array)
                    st.session_state['landmarks_px'] = landmarks_px
            landmarks_px = st.session_state['landmarks_px']
            img_annotated = draw_landmarks_on_image(img_array, landmarks_px)
            st.session_state['img_annotated'] = img_annotated

            col_orig, col_annot = st.columns(2)
            with col_orig:
                st.markdown('<div class="section-title">Radio originale</div>',
                            unsafe_allow_html=True)
                st.image(img_array, use_container_width=True)
            with col_annot:
                st.markdown('<div class="section-title">19 Landmarks détectés</div>',
                            unsafe_allow_html=True)
                st.image(img_annotated, use_container_width=True)

            # ══════════════════════════════════════════════════════════
            # CORRECTION MANUELLE DES LANDMARKS
            # ══════════════════════════════════════════════════════════
            from streamlit_image_coordinates import streamlit_image_coordinates

            st.markdown('<div class="section-title">✏️ Correction manuelle</div>',
                        unsafe_allow_html=True)

            col_toggle, col_reset = st.columns([2, 1])
            with col_toggle:
                correction_mode = st.toggle('Activer le mode correction', value=False)
            with col_reset:
                if st.button('↺ Réinitialiser aux prédictions IA'):
                    st.session_state['landmarks_px'] = predict_landmarks(model, img_array)
                    st.session_state.pop('selected_landmark', None)
                    st.rerun()

            if correction_mode:
                landmarks_px = st.session_state['landmarks_px']

                selected_idx = st.selectbox(
                    'Point à corriger',
                    options=list(range(19)),
                    format_func=lambda i: f'{i+1}. {LANDMARK_NAMES[i]}',
                    index=st.session_state.get('selected_landmark', 0),
                    key='landmark_selector'
                )
                st.session_state['selected_landmark'] = selected_idx

                st.caption(
                    f'Point sélectionné : **{LANDMARK_NAMES[selected_idx]}** — '
                    f"clique sur l'image ci-dessous à l'endroit correct."
                )

                img_highlight = draw_landmarks_on_image(img_array, landmarks_px)
                hx = int(landmarks_px[selected_idx][0])
                hy = int(landmarks_px[selected_idx][1])
                cv2.circle(img_highlight, (hx, hy), 14, (255, 255, 0), 3)

                click = streamlit_image_coordinates(
                    img_highlight, key=f'click_correction_{selected_idx}'
                )

                if click is not None:
                    disp_w = click.get('width', img_w)
                    disp_h = click.get('height', img_h)
                    new_x = click['x'] * (img_w / disp_w)
                    new_y = click['y'] * (img_h / disp_h)

                    if (abs(new_x - landmarks_px[selected_idx][0]) > 1 or
                            abs(new_y - landmarks_px[selected_idx][1]) > 1):
                        landmarks_px[selected_idx] = [new_x, new_y]
                        st.session_state['landmarks_px'] = landmarks_px
                        st.session_state['img_annotated'] = draw_landmarks_on_image(
                            img_array, landmarks_px)
                        st.rerun()

                st.info('💡 Les mesures de Steiner, le diagnostic et le rapport '
                        'se recalculent automatiquement avec les points corrigés.')

        else:
            st.markdown('<div class="section-title">Radio chargée</div>',
                        unsafe_allow_html=True)
            st.image(img_array, use_container_width=True)
            st.warning('⚠️ Modèle non chargé — Vérifiez que best_model_heatmap.pth existe dans models/checkpoints/')

            # Mode démo — coordonnées simulées
            st.info('Mode démo : coordonnées simulées pour illustration.')
            demo_coords = np.array([
                [img_w*0.50, img_h*0.17], [img_w*0.54, img_h*0.15],
                [img_w*0.65, img_h*0.19], [img_w*0.41, img_h*0.19],
                [img_w*0.57, img_h*0.36], [img_w*0.56, img_h*0.33],
                [img_w*0.46, img_h*0.33], [img_w*0.58, img_h*0.40],
                [img_w*0.57, img_h*0.44], [img_w*0.57, img_h*0.42],
                [img_w*0.56, img_h*0.46], [img_w*0.54, img_h*0.46],
                [img_w*0.53, img_h*0.52], [img_w*0.52, img_h*0.56],
                [img_w*0.51, img_h*0.58], [img_w*0.40, img_h*0.50],
                [img_w*0.52, img_h*0.48], [img_w*0.39, img_h*0.21],
                [img_w*0.36, img_h*0.23],
            ])
            img_demo = draw_landmarks_on_image(img_array, demo_coords)
            st.image(img_demo, use_container_width=True, caption='Visualisation démo')
            st.session_state['landmarks_px'] = demo_coords
            st.session_state['img_annotated'] = img_demo

    with col_info:
        st.markdown('<div class="section-title">Informations</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:0.5rem;">
            <div class="metric-label">Patient</div>
            <div style="font-weight:600; color:#F1F5F9; font-size:0.95rem;">
                {nom_patient or 'Non renseigné'}
            </div>
            <div style="font-size:0.8rem; color:#64748B;">{age_patient or '—'}</div>
        </div>
        <div class="metric-card" style="margin-bottom:0.5rem;">
            <div class="metric-label">Dimensions image</div>
            <div style="font-family:'JetBrains Mono'; font-size:0.9rem; color:#F1F5F9;">
                {img_w} × {img_h} px
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Résolution</div>
            <div style="font-family:'JetBrains Mono'; font-size:0.9rem; color:#F1F5F9;">
                {scale} mm/px
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Légende couleurs
        st.markdown('<div class="section-title" style="margin-top:1rem;">Légende</div>',
                    unsafe_allow_html=True)
        for label, color in [
            ('Points médians', '#FF6B6B'),
            ('Points bilatéraux', '#4ECDC4'),
            ('Incisives', '#FFE66D'),
        ]:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.5rem;
                        padding:0.4rem 0; font-size:0.8rem; color:#94A3B8;">
                <div style="width:10px; height:10px; border-radius:50%;
                            background:{color}; flex-shrink:0;"></div>
                {label}
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ONGLET 2 — MESURES DE STEINER
# ══════════════════════════════════════════════════════════════

with tab2:
    if 'landmarks_px' not in st.session_state:
        st.info('Chargez d\'abord une radiographie dans l\'onglet Radiographie.')
        st.stop()

    landmarks_px = st.session_state['landmarks_px']
    coords_dict  = {name: (float(landmarks_px[i,0]), float(landmarks_px[i,1]))
                    for i, name in enumerate(LANDMARK_NAMES)}
    
    results   = compute_steiner_analysis(coords_dict, img_w, img_h, scale)
    formatted = format_results(results)
    st.session_state['results']   = results
    st.session_state['formatted'] = formatted

    # Métriques globales rapides
    n_normal = sum(1 for f in formatted if f['statut'] == 'NORMAL')
    n_hors   = len(formatted) - n_normal

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mesures calculées</div>
            <div class="metric-value">{len(formatted)}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Dans la norme</div>
            <div class="metric-value" style="color:#10B981;">{n_normal}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Hors norme</div>
            <div class="metric-value" style="color:#F59E0B;">{n_hors}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        anb = results.get('ANB', 0)
        classe_color = '#10B981' if abs(anb-2) <= 2 else '#F59E0B' if anb > 0 else '#EF4444'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ANB</div>
            <div class="metric-value" style="color:{classe_color};">{anb:.1f}°</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Sections
    sections = {
    'Diagnostic Squelettique — Antéro-postérieur':
        ['SNA', 'SNB', 'ANB'],
    'Diagnostic Squelettique — Vertical':
        ['Go_Gn_SN', 'SE', 'SL'],
    'Diagnostic Dento-squelettique':
        ['I_NA_distance', 'i_NB_distance', 'Occ_SN'],
    }

    fm_dict = {f['key']: f for f in formatted}

    for section_name, keys in sections.items():
        st.markdown(f'<div class="section-title">{section_name}</div>',
                    unsafe_allow_html=True)
        rows_html = ''
        for key in keys:
            if key in fm_dict:
                rows_html += render_measure_row(fm_dict[key])
        st.markdown(rows_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ONGLET 3 — DIAGNOSTIC CLINIQUE
# ══════════════════════════════════════════════════════════════

with tab3:
    if 'results' not in st.session_state:
        st.info('Consultez d\'abord l\'onglet Mesures de Steiner.')
        st.stop()

    results    = st.session_state['results']
    diagnostic = generer_diagnostic_complet(results)
    diag_dict  = diagnostic_to_dict(diagnostic)
    st.session_state['diagnostic'] = diag_dict

    # Résumé global
    emoji_map = {'✅':'#10B981','🟡':'#F59E0B','🟠':'#F97316','🔴':'#EF4444'}
    accent    = emoji_map.get(diagnostic.emoji_global, '#3B82F6')

    st.markdown(f"""
    <div style="background:linear-gradient(135deg, {accent}15, {accent}08);
                border:1px solid {accent}30; border-radius:{12}px;
                padding:1.5rem; margin-bottom:1.5rem;">
        <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
            <div style="font-size:2.5rem;">{diagnostic.emoji_global}</div>
            <div style="flex:1;">
                <div style="font-size:1.1rem; font-weight:700; color:#F1F5F9; margin-bottom:0.3rem;">
                    {diagnostic.resume}
                </div>
                <div style="font-size:0.8rem; color:#94A3B8;">
                    Priorité de traitement :
                    <strong style="color:{accent};">{diagnostic.priorite_traitement}</strong>
                </div>
            </div>
        </div>
        <div style="display:flex; gap:0.5rem; margin-top:1rem; flex-wrap:wrap;">
            <span class="classe-badge classe-{'I' if 'I' in diagnostic.classe_squelettique and 'II' not in diagnostic.classe_squelettique and 'III' not in diagnostic.classe_squelettique else 'II' if 'II' in diagnostic.classe_squelettique and 'III' not in diagnostic.classe_squelettique else 'III'}">
                🎯 {diagnostic.classe_squelettique}
            </span>
            <span class="classe-badge {'classe-II' if 'Hyper' in diagnostic.divergence else 'classe-III' if 'Hypo' in diagnostic.divergence else 'classe-I'}">
                📐 {diagnostic.divergence}
            </span>
            <span class="classe-badge {'classe-II' if diagnostic.maxillaire != 'Normal' else 'classe-I'}">
                🦷 Maxillaire {diagnostic.maxillaire}
            </span>
            <span class="classe-badge {'classe-II' if diagnostic.mandibule != 'Normal' else 'classe-I'}">
                🦷 Mandibule {diagnostic.mandibule}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sections de diagnostic
    diag_sections = [
        ('📋 Diagnostic Squelettique',      diag_dict['squelettique']),
        ('🦷 Diagnostic Dento-squelettique', diag_dict['dento_squelettique']),
        ('🔗 Diagnostic Dento-dentaire',     diag_dict['dento_dentaire']),
        ('💄 Diagnostic Esthétique',         diag_dict['esthetique']),
    ]

    for section_name, items in diag_sections:
        if not items:
            continue
        st.markdown(f'<div class="section-title">{section_name}</div>',
                    unsafe_allow_html=True)
        html = ''.join(render_diag_item(item) for item in items)
        st.markdown(html, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div style="margin-top:2rem; padding:1rem 1.25rem;
                background:#0F1629; border:1px solid #1E293B;
                border-radius:8px; font-size:0.78rem; color:#475569;
                border-left:3px solid #3B82F6;">
        ⚠️ Ce rapport est généré automatiquement par CéphaloAI à titre d'aide au diagnostic.
        Il ne remplace en aucun cas l'examen clinique et le jugement du praticien.
        Validation clinique obligatoire avant tout plan de traitement.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ONGLET 4 — RAPPORT EXPORTABLE
# ══════════════════════════════════════════════════════════════

with tab4:
    if 'diagnostic' not in st.session_state:
        st.info('Consultez d\'abord l\'onglet Diagnostic.')
        st.stop()

    diag_dict = st.session_state['diagnostic']

    st.markdown('<div class="section-title">Export du rapport</div>',
                unsafe_allow_html=True)

    col_pdf, col_img = st.columns(2)

    with col_pdf:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Rapport PDF</div>
            <div style="color:#F1F5F9; font-size:0.875rem; margin:0.5rem 0;">
                Rapport complet avec radio annotée, tableau des mesures
                et diagnostic clinique.
            </div>
        </div>""", unsafe_allow_html=True)

        if st.button('📄 Générer le rapport PDF', use_container_width=True):
            try:
                sys.path.insert(0, str(ROOT / 'src'))
                from report import generer_rapport_pdf
                pdf_bytes = generer_rapport_pdf(
                    nom_patient   = nom_patient or 'Patient',
                    age           = age_patient or '—',
                    img_array     = img_array,
                    img_annotated = st.session_state.get('img_annotated', img_array),
                    formatted     = st.session_state.get('formatted', []),
                    diag_dict     = diag_dict,
                )
                st.download_button(
                    label    = '⬇️ Télécharger le PDF',
                    data     = pdf_bytes,
                    file_name= f'CephaloAI_{nom_patient or "patient"}.pdf',
                    mime     = 'application/pdf',
                    use_container_width=True
                )
                st.success('✅ Rapport généré !')
            except Exception as e:
                st.error(f'Erreur PDF : {e}')
                st.info('Assurez-vous que report.py est dans src/')

    with col_img:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Historique patient</div>
            <div style="color:#F1F5F9; font-size:0.875rem; margin:0.5rem 0;">
                Enregistre cette visite pour pouvoir la comparer à une
                prochaine radio du même patient (onglet Suivi).
            </div>
        </div>""", unsafe_allow_html=True)

        if st.button('💾 Enregistrer cette visite', use_container_width=True):
            save_visit(
                nom_patient, age_patient,
                st.session_state.get('img_annotated', img_array),
                st.session_state.get('landmarks_px'),
                st.session_state.get('results', {})
            )
            st.success('✅ Visite enregistrée dans l\'historique !')

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Image annotée</div>
            <div style="color:#F1F5F9; font-size:0.875rem; margin:0.5rem 0;">
                Radio avec les 19 landmarks et lignes de Steiner tracés.
            </div>
        </div>""", unsafe_allow_html=True)

        if 'img_annotated' in st.session_state:
            img_ann  = st.session_state['img_annotated']
            img_pil2 = Image.fromarray(img_ann)
            buf = io.BytesIO()
            img_pil2.save(buf, format='PNG')
            st.download_button(
                label    = '⬇️ Télécharger la radio annotée',
                data     = buf.getvalue(),
                file_name= f'radio_annotee_{nom_patient or "patient"}.png',
                mime     = 'image/png',
                use_container_width=True
            )

    # Aperçu résumé
    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Aperçu du rapport</div>',
                unsafe_allow_html=True)

    from datetime import date
    today = date.today().strftime('%d/%m/%Y')

    st.markdown(f"""
    <div style="background:#0F1629; border:1px solid #1E293B;
                border-radius:12px; padding:2rem; font-family:'Inter',sans-serif;">
        <div style="display:flex; justify-content:space-between; align-items:start;
                    margin-bottom:1.5rem; padding-bottom:1rem;
                    border-bottom:1px solid #1E293B;">
            <div>
                <div style="font-size:1.3rem; font-weight:700;
                            background:linear-gradient(135deg,#3B82F6,#06B6D4);
                            -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                    CéphaloAI
                </div>
                <div style="font-size:0.8rem; color:#64748B;">
                    Cabinet Dr. Redouane Chiguer
                </div>
            </div>
            <div style="text-align:right; font-size:0.8rem; color:#64748B;">
                <div>{today}</div>
                <div>Analyse de Steiner 1952</div>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem;">
            <div>
                <div style="font-size:0.7rem; text-transform:uppercase;
                            letter-spacing:0.08em; color:#475569; margin-bottom:0.3rem;">
                    Patient
                </div>
                <div style="font-weight:600; color:#F1F5F9;">
                    {nom_patient or 'Non renseigné'}
                </div>
                <div style="font-size:0.8rem; color:#64748B;">{age_patient or '—'}</div>
            </div>
            <div>
                <div style="font-size:0.7rem; text-transform:uppercase;
                            letter-spacing:0.08em; color:#475569; margin-bottom:0.3rem;">
                    Diagnostic principal
                </div>
                <div style="font-weight:600; color:#F1F5F9;">
                    {diag_dict.get('classe_squelettique','—')}
                </div>
                <div style="font-size:0.8rem; color:#64748B;">
                    {diag_dict.get('divergence','—')}
                </div>
            </div>
        </div>
        <div style="padding:1rem; background:#141B2D; border-radius:8px;
                    border-left:3px solid #3B82F6;">
            <div style="font-size:0.7rem; text-transform:uppercase;
                        letter-spacing:0.08em; color:#475569; margin-bottom:0.3rem;">
                Résumé clinique
            </div>
            <div style="font-size:0.875rem; color:#F1F5F9; line-height:1.6;">
                {diag_dict.get('resume','—')}
            </div>
            <div style="margin-top:0.5rem; font-size:0.8rem; color:#94A3B8;">
                Priorité : <strong style="color:#3B82F6;">
                {diag_dict.get('priorite','—')}</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ONGLET 5 — SUIVI / HISTORIQUE PATIENT
# ══════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<div class="section-title">Historique du patient</div>',
                unsafe_allow_html=True)

    patients = list_patients()

    if not patients:
        st.info('Aucune visite enregistrée pour l\'instant. '
                'Enregistre une visite depuis l\'onglet Rapport pour commencer '
                'un suivi.')
    else:
        slug_to_name = {}
        for slug in patients:
            visits = list_visits(slug)
            if visits:
                slug_to_name[slug] = visits[0]['patient_name']

        selected_slug = st.selectbox(
            'Patient',
            options=patients,
            format_func=lambda s: slug_to_name.get(s, s)
        )
        visits = list_visits(selected_slug)

        if len(visits) < 2:
            st.info('Une seule visite enregistrée pour ce patient — '
                    'reviens ici après une deuxième visite pour comparer '
                    'l\'évolution.')
            v = visits[0]
            img_bytes = base64.b64decode(v['image_b64'])
            st.markdown(f'<div class="section-title">{v["date"]}</div>',
                        unsafe_allow_html=True)
            st.image(img_bytes, use_container_width=True)
        else:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                idx1 = st.selectbox(
                    'Visite 1 (référence)',
                    options=list(range(len(visits))),
                    format_func=lambda i: visits[i]['date'],
                    index=0, key='visit1_select'
                )
            with col_v2:
                idx2 = st.selectbox(
                    'Visite 2 (comparaison)',
                    options=list(range(len(visits))),
                    format_func=lambda i: visits[i]['date'],
                    index=len(visits) - 1, key='visit2_select'
                )

            v1, v2 = visits[idx1], visits[idx2]

            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.markdown(f'<div class="section-title">{v1["date"]}</div>',
                            unsafe_allow_html=True)
                st.image(base64.b64decode(v1['image_b64']),
                        use_container_width=True)
            with col_img2:
                st.markdown(f'<div class="section-title">{v2["date"]}</div>',
                            unsafe_allow_html=True)
                st.image(base64.b64decode(v2['image_b64']),
                        use_container_width=True)

            st.markdown('<div class="section-title">Évolution des mesures</div>',
                        unsafe_allow_html=True)

            keys_common = [
                k for k in v1['results']
                if k in v2['results'] and isinstance(v1['results'][k], (int, float))
                and isinstance(v2['results'][k], (int, float))
            ]

            rows_html = ''
            for key in sorted(keys_common):
                val1 = v1['results'][key]
                val2 = v2['results'][key]
                delta = val2 - val1
                if delta > 0.05:
                    arrow, color = '↑', '#F59E0B'
                elif delta < -0.05:
                    arrow, color = '↓', '#3B82F6'
                else:
                    arrow, color = '→', '#94A3B8'
                rows_html += f"""
                <div class="measure-row">
                    <span class="measure-name">{key}</span>
                    <span class="measure-val">{val1:.1f} → {val2:.1f}</span>
                    <span class="measure-norm" style="color:{color};">
                        Δ {delta:+.1f}  {arrow}
                    </span>
                </div>"""
            st.markdown(rows_html, unsafe_allow_html=True)