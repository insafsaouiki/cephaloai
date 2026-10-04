"""
CéphaloAI — report.py
Insaf Saouiki — EMSI — Stage Cabinet Dr. Redouane Chiguer

Génération du rapport PDF professionnel d'analyse céphalométrique.
"""

import io
from datetime import date
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import (
    HexColor, white, black
)
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, Image as RLImage, PageBreak
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas


# ── Couleurs ───────────────────────────────────────────────
C_BLUE      = HexColor('#1A3C6E')
C_TEAL      = HexColor('#0E7C86')
C_GREEN     = HexColor('#10B981')
C_AMBER     = HexColor('#F59E0B')
C_ORANGE    = HexColor('#F97316')
C_RED       = HexColor('#EF4444')
C_GRAY      = HexColor('#4A4A4A')
C_LGRAY     = HexColor('#F4F7FB')
C_BORDER    = HexColor('#E2E8F0')
C_WHITE     = white
C_BLACK     = black


# ── Styles de texte ────────────────────────────────────────
def make_styles():
    return {
        'title': ParagraphStyle(
            'title', fontName='Helvetica-Bold',
            fontSize=22, textColor=C_BLUE,
            spaceAfter=4, alignment=TA_LEFT
        ),
        'subtitle': ParagraphStyle(
            'subtitle', fontName='Helvetica',
            fontSize=11, textColor=C_TEAL,
            spaceAfter=2, alignment=TA_LEFT
        ),
        'section': ParagraphStyle(
            'section', fontName='Helvetica-Bold',
            fontSize=11, textColor=C_BLUE,
            spaceBefore=14, spaceAfter=6
        ),
        'body': ParagraphStyle(
            'body', fontName='Helvetica',
            fontSize=9, textColor=C_GRAY,
            spaceAfter=4, leading=14
        ),
        'small': ParagraphStyle(
            'small', fontName='Helvetica',
            fontSize=7.5, textColor=HexColor('#64748B'),
            spaceAfter=2
        ),
        'disclaimer': ParagraphStyle(
            'disclaimer', fontName='Helvetica-Oblique',
            fontSize=7.5, textColor=HexColor('#94A3B8'),
            spaceAfter=2, alignment=TA_CENTER
        ),
        'bold': ParagraphStyle(
            'bold', fontName='Helvetica-Bold',
            fontSize=9, textColor=C_GRAY,
            spaceAfter=4
        ),
        'center': ParagraphStyle(
            'center', fontName='Helvetica',
            fontSize=9, textColor=C_GRAY,
            alignment=TA_CENTER
        ),
    }


# ── Convertir image numpy → ReportLab Image ───────────────
def numpy_to_rl_image(img_array, max_width, max_height):
    """Convertit un array numpy en image ReportLab."""
    if img_array is None:
        return None
    try:
        if img_array.ndim == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img_array.astype(np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)

        # Calculer dimensions proportionnelles
        w, h = pil_img.size
        ratio = min(max_width/w, max_height/h)
        return RLImage(buf, width=w*ratio, height=h*ratio)
    except Exception:
        return None


# ── Couleur selon statut ───────────────────────────────────
def statut_color(statut):
    return {
        'NORMAL': C_GREEN,
        'LEGER' : C_AMBER,
        'MODERE': C_ORANGE,
        'SEVERE': C_RED,
    }.get(statut, C_GRAY)


def statut_label(statut):
    return {
        'NORMAL': '✓ Normal',
        'LEGER' : '⚠ Léger',
        'MODERE': '⚠ Modéré',
        'SEVERE': '✗ Sévère',
    }.get(statut, statut)


# ══════════════════════════════════════════════════════════
# FONCTION PRINCIPALE
# ══════════════════════════════════════════════════════════

def generer_rapport_pdf(nom_patient, age, img_array,
                        img_annotated, formatted, diag_dict):
    """
    Génère le rapport PDF complet d'analyse céphalométrique.

    Args:
        nom_patient   : str — nom du patient
        age           : str — âge du patient
        img_array     : numpy array — radio originale
        img_annotated : numpy array — radio avec landmarks
        formatted     : list — mesures formatées (sortie de geometry.py)
        diag_dict     : dict — diagnostic (sortie de interpretation.py)

    Returns:
        bytes — contenu du PDF
    """
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf,
        pagesize      = A4,
        leftMargin    = 1.8*cm,
        rightMargin   = 1.8*cm,
        topMargin     = 1.5*cm,
        bottomMargin  = 1.5*cm,
    )
    styles  = make_styles()
    story   = []
    W       = A4[0] - 3.6*cm   # largeur utile
    today   = date.today().strftime('%d/%m/%Y')

    # ══════════════════════════════════════════════════════
    # EN-TÊTE
    # ══════════════════════════════════════════════════════

    # Bandeau bleu header
    header_data = [[
        Paragraph('🦷 CéphaloAI', ParagraphStyle(
            'h', fontName='Helvetica-Bold', fontSize=18,
            textColor=C_WHITE)),
        Paragraph(
            'Cabinet Dentaire Dr. Redouane Chiguer<br/>'
            '<font size="8">Analyse Céphalométrique de Steiner — 1952</font>',
            ParagraphStyle('hs', fontName='Helvetica',
                           fontSize=10, textColor=C_WHITE,
                           alignment=TA_RIGHT)
        )
    ]]
    header_tbl = Table(header_data, colWidths=[W*0.5, W*0.5])
    header_tbl.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,-1), C_BLUE),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING',(0,0), (-1,-1), 14),
        ('TOPPADDING',  (0,0), (-1,-1), 12),
        ('BOTTOMPADDING',(0,0),(-1,-1), 12),
        ('ROUNDEDCORNERS', [6]),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── Infos patient ──────────────────────────────────────
    info_data = [
        [
            Paragraph('<b>Patient</b>', styles['small']),
            Paragraph('<b>Date d\'analyse</b>', styles['small']),
            Paragraph('<b>Diagnostic principal</b>', styles['small']),
            Paragraph('<b>Priorité</b>', styles['small']),
        ],
        [
            Paragraph(f'{nom_patient}<br/><font size="7">{age}</font>',
                      styles['body']),
            Paragraph(today, styles['body']),
            Paragraph(diag_dict.get('classe_squelettique', '—'),
                      styles['body']),
            Paragraph(diag_dict.get('priorite', '—'), styles['body']),
        ]
    ]
    info_tbl = Table(info_data, colWidths=[W*0.28, W*0.18, W*0.28, W*0.26])
    info_tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,0), C_LGRAY),
        ('BACKGROUND',   (0,1), (-1,1), C_WHITE),
        ('BOX',          (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID',    (0,0), (-1,-1), 0.3, C_BORDER),
        ('LEFTPADDING',  (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING',   (0,0), (-1,-1), 6),
        ('BOTTOMPADDING',(0,0), (-1,-1), 6),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── Résumé clinique ────────────────────────────────────
    resume = diag_dict.get('resume', '—')
    emoji  = diag_dict.get('emoji', '🔵')
    resume_data = [[
        Paragraph(f'{emoji} <b>Résumé :</b> {resume}', styles['body'])
    ]]
    resume_tbl = Table(resume_data, colWidths=[W])
    resume_tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), HexColor('#EFF6FF')),
        ('BOX',          (0,0), (-1,-1), 1, C_TEAL),
        ('LEFTPADDING',  (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING',   (0,0), (-1,-1), 8),
        ('BOTTOMPADDING',(0,0), (-1,-1), 8),
    ]))
    story.append(resume_tbl)
    story.append(Spacer(1, 0.5*cm))

    # ══════════════════════════════════════════════════════
    # IMAGES — Radio originale + annotée
    # ══════════════════════════════════════════════════════

    story.append(HRFlowable(width=W, thickness=0.5,
                             color=C_BORDER, spaceAfter=8))
    story.append(Paragraph('Téléradiographie Latérale', styles['section']))

    img_w  = W * 0.48
    img_h  = 6.5 * cm
    rl_orig  = numpy_to_rl_image(img_array,     img_w, img_h)
    rl_annot = numpy_to_rl_image(img_annotated, img_w, img_h)

    if rl_orig and rl_annot:
        img_data = [[
            Table([[Paragraph('Radio originale', styles['small'])],
                   [rl_orig]],
                  colWidths=[img_w]),
            Table([[Paragraph('19 Landmarks détectés', styles['small'])],
                   [rl_annot]],
                  colWidths=[img_w]),
        ]]
        img_tbl = Table(img_data, colWidths=[W*0.5, W*0.5])
        img_tbl.setStyle(TableStyle([
            ('VALIGN',      (0,0), (-1,-1), 'TOP'),
            ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING',(0,0), (-1,-1), 4),
        ]))
        story.append(img_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ══════════════════════════════════════════════════════
    # TABLEAU DES MESURES
    # ══════════════════════════════════════════════════════

    story.append(HRFlowable(width=W, thickness=0.5,
                             color=C_BORDER, spaceAfter=8))
    story.append(Paragraph('Mesures de l\'Analyse de Steiner',
                            styles['section']))

    # Sections des mesures
    sections_map = {
        'Diagnostic Squelettique — Antéro-postérieur':
            ['SNA', 'SNB', 'ANB', 'SND'],
        'Diagnostic Squelettique — Vertical':
            ['Go_Gn_SN', 'SE', 'SL'],
        'Diagnostic Dento-squelettique':
            ['I_NA_angle', 'I_NA_distance', 'i_NB_angle',
             'i_NB_distance', 'Occ_SN'],
        'Diagnostic Dento-dentaire':
            ['inter_incisif'],
    }

    fm_dict = {f['key']: f for f in formatted} if formatted else {}

    for section_name, keys in sections_map.items():
        section_items = [fm_dict[k] for k in keys if k in fm_dict]
        if not section_items:
            continue

        # Titre de section
        sec_data = [[Paragraph(section_name, ParagraphStyle(
            'sec', fontName='Helvetica-Bold', fontSize=8,
            textColor=C_WHITE))]]
        sec_tbl = Table(sec_data, colWidths=[W])
        sec_tbl.setStyle(TableStyle([
            ('BACKGROUND',   (0,0), (-1,-1), C_TEAL),
            ('LEFTPADDING',  (0,0), (-1,-1), 8),
            ('TOPPADDING',   (0,0), (-1,-1), 4),
            ('BOTTOMPADDING',(0,0), (-1,-1), 4),
        ]))
        story.append(sec_tbl)

        # Lignes de mesures
        col_w = [W*0.30, W*0.14, W*0.18, W*0.20, W*0.18]
        rows  = [[
            Paragraph('<b>Mesure</b>', styles['small']),
            Paragraph('<b>Valeur</b>', styles['small']),
            Paragraph('<b>Norme</b>',  styles['small']),
            Paragraph('<b>Statut</b>', styles['small']),
            Paragraph('<b>Interprétation</b>', styles['small']),
        ]]

        for f in section_items:
            color  = statut_color(f['statut'])
            label  = statut_label(f['statut'])
            rows.append([
                Paragraph(f['label'], styles['body']),
                Paragraph(
                    f"<b>{f['valeur']}{f['unite']}</b>",
                    ParagraphStyle('v', fontName='Helvetica-Bold',
                                   fontSize=9, textColor=color)
                ),
                Paragraph(f['norme'], styles['small']),
                Paragraph(label, ParagraphStyle(
                    'st', fontName='Helvetica-Bold',
                    fontSize=8, textColor=color)),
                Paragraph(f['interpretation'], styles['small']),
            ])

        mes_tbl = Table(rows, colWidths=col_w)
        style_list = [
            ('BACKGROUND',   (0,0), (-1,0),  C_LGRAY),
            ('BOX',          (0,0), (-1,-1),  0.5, C_BORDER),
            ('INNERGRID',    (0,0), (-1,-1),  0.3, C_BORDER),
            ('LEFTPADDING',  (0,0), (-1,-1),  6),
            ('RIGHTPADDING', (0,0), (-1,-1),  6),
            ('TOPPADDING',   (0,0), (-1,-1),  4),
            ('BOTTOMPADDING',(0,0), (-1,-1),  4),
            ('VALIGN',       (0,0), (-1,-1),  'MIDDLE'),
        ]
        for i in range(1, len(rows)):
            if i % 2 == 0:
                style_list.append(
                    ('BACKGROUND', (0,i), (-1,i), C_LGRAY)
                )
        mes_tbl.setStyle(TableStyle(style_list))
        story.append(mes_tbl)
        story.append(Spacer(1, 0.25*cm))

    # ══════════════════════════════════════════════════════
    # DIAGNOSTIC CLINIQUE
    # ══════════════════════════════════════════════════════

    story.append(PageBreak())
    story.append(HRFlowable(width=W, thickness=0.5,
                             color=C_BORDER, spaceAfter=8))
    story.append(Paragraph('Diagnostic Clinique Complet',
                            styles['section']))

    # Badges classification
    badges_data = [[
        Paragraph(
            f"<b>Classe squelettique :</b> "
            f"{diag_dict.get('classe_squelettique','—')}",
            styles['body']),
        Paragraph(
            f"<b>Divergence :</b> "
            f"{diag_dict.get('divergence','—')}",
            styles['body']),
        Paragraph(
            f"<b>Maxillaire :</b> "
            f"{diag_dict.get('maxillaire','—')}",
            styles['body']),
        Paragraph(
            f"<b>Mandibule :</b> "
            f"{diag_dict.get('mandibule','—')}",
            styles['body']),
    ]]
    badges_tbl = Table(badges_data, colWidths=[W/4]*4)
    badges_tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), HexColor('#EFF6FF')),
        ('BOX',          (0,0), (-1,-1), 0.5, C_BLUE),
        ('INNERGRID',    (0,0), (-1,-1), 0.3, C_BORDER),
        ('LEFTPADDING',  (0,0), (-1,-1), 8),
        ('TOPPADDING',   (0,0), (-1,-1), 6),
        ('BOTTOMPADDING',(0,0), (-1,-1), 6),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(badges_tbl)
    story.append(Spacer(1, 0.3*cm))

    # Sections du diagnostic
    diag_sections = [
        ('Diagnostic Squelettique',      diag_dict.get('squelettique', [])),
        ('Diagnostic Dento-squelettique',diag_dict.get('dento_squelettique', [])),
        ('Diagnostic Dento-dentaire',    diag_dict.get('dento_dentaire', [])),
        ('Diagnostic Esthétique',        diag_dict.get('esthetique', [])),
    ]

    for section_name, items in diag_sections:
        if not items:
            continue

        story.append(Paragraph(section_name, ParagraphStyle(
            'dsec', fontName='Helvetica-Bold', fontSize=9,
            textColor=C_BLUE, spaceBefore=10, spaceAfter=4
        )))

        for item in items:
            color  = statut_color(item.get('statut', 'NORMAL'))
            diag_data = [[
                Paragraph(
                    f"• {item.get('diagnostic','—')}",
                    ParagraphStyle('di', fontName='Helvetica-Bold',
                                   fontSize=8.5, textColor=color)
                ),
            ], [
                Paragraph(
                    f"  → {item.get('conseil','—')}",
                    ParagraphStyle('dc', fontName='Helvetica-Oblique',
                                   fontSize=8, textColor=C_GRAY,
                                   leftIndent=10)
                ),
            ]]
            diag_tbl = Table(diag_data, colWidths=[W])
            diag_tbl.setStyle(TableStyle([
                ('LEFTPADDING',  (0,0), (-1,-1), 10),
                ('TOPPADDING',   (0,0), (-1,-1), 2),
                ('BOTTOMPADDING',(0,0), (-1,-1), 2),
                ('LINEAFTER',    (0,0), (0,-1), 2, color),
            ]))
            story.append(diag_tbl)

        story.append(Spacer(1, 0.2*cm))

    # ══════════════════════════════════════════════════════
    # PIED DE PAGE — Disclaimer
    # ══════════════════════════════════════════════════════

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width=W, thickness=0.5,
                             color=C_BORDER, spaceAfter=8))

    disclaimer_data = [[
        Paragraph(
            '⚠️ Ce rapport est généré automatiquement par CéphaloAI '
            'à titre d\'aide au diagnostic orthodontique. '
            'Il ne remplace en aucun cas l\'examen clinique et le jugement '
            'du praticien. Une validation clinique est obligatoire avant '
            'tout plan de traitement. — Analyse de Steiner 1952 — '
            f'Généré le {today}',
            styles['disclaimer']
        )
    ]]
    disc_tbl = Table(disclaimer_data, colWidths=[W])
    disc_tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), C_LGRAY),
        ('BOX',          (0,0), (-1,-1), 0.5, C_BORDER),
        ('LEFTPADDING',  (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING',   (0,0), (-1,-1), 8),
        ('BOTTOMPADDING',(0,0), (-1,-1), 8),
    ]))
    story.append(disc_tbl)

    # ── Générer le PDF ─────────────────────────────────────
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ── Test rapide ────────────────────────────────────────────
if __name__ == '__main__':
    print('🧪 Test report.py\n')

    # Créer une image test
    img_test = np.ones((800, 600, 3), dtype=np.uint8) * 30
    cv2.putText(img_test, 'Radio Test', (150, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)

    formatted_test = [
        {'key':'SNA','label':'Angle SNA','valeur':85.0,'unite':'°',
         'norme':'82° ± 2°','statut':'LEGER','interpretation':'Prognathie légère'},
        {'key':'SNB','label':'Angle SNB','valeur':78.0,'unite':'°',
         'norme':'80° ± 2°','statut':'LEGER','interpretation':'Rétrognathie légère'},
        {'key':'ANB','label':'Angle ANB','valeur':7.0,'unite':'°',
         'norme':'2° ± 2°','statut':'MODERE','interpretation':'Classe II modérée'},
        {'key':'Go_Gn_SN','label':'Angle Go-Gn/SN','valeur':38.0,'unite':'°',
         'norme':'32° ± 5°','statut':'LEGER','interpretation':'Hyperdivergence légère'},
        {'key':'inter_incisif','label':'Angle inter-incisif','valeur':120.0,'unite':'°',
         'norme':'131° ± 5°','statut':'MODERE','interpretation':'Biproalvéolie'},
    ]

    diag_test = {
        'classe_squelettique' : 'Classe II',
        'divergence'          : 'Hyperdivergent',
        'maxillaire'          : 'Prognathique',
        'mandibule'           : 'Normal',
        'resume'              : 'Classe II squelettique modérée — Hyperdivergence',
        'priorite'            : 'Recommandée',
        'emoji'               : '🟠',
        'squelettique'        : [
            {'mesure':'SNA','valeur':85.0,'norme':'82° ± 2°',
             'statut':'LEGER','diagnostic':'Prognathie maxillaire légère',
             'conseil':'Recul maxillaire selon croissance'}
        ],
        'dento_squelettique'  : [],
        'dento_dentaire'      : [
            {'mesure':'inter_incisif','valeur':120.0,'norme':'131° ± 5°',
             'statut':'MODERE','diagnostic':'Biproalvéolie',
             'conseil':'Extractions et recul des incisives'}
        ],
        'esthetique'          : [],
    }

    pdf_bytes = generer_rapport_pdf(
        nom_patient   = 'Patient Test',
        age           = '14 ans',
        img_array     = img_test,
        img_annotated = img_test,
        formatted     = formatted_test,
        diag_dict     = diag_test,
    )

    output_path = Path('models/checkpoints/test_rapport.pdf')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pdf_bytes)

    print(f'✅ PDF généré : {output_path}')
    print(f'   Taille : {len(pdf_bytes)/1024:.1f} KB')
    print()
    print('='*45)
    print('  ✅ REPORT OK — Prêt pour app.py !')
    print('='*45)
