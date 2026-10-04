"""
CéphaloAI — geometry.py
Insaf Saouiki — EMSI — Stage Cabinet Dr. Redouane Chiguer

Ordre RÉEL ISBI 2015 (vérifié) :
1=S, 2=Na, 3=Or, 4=Po, 5=A, 6=B, 7=Pog, 8=Me, 9=Gn, 10=Go,
11=L1 (pointe incisive inf.), 12=U1 (pointe incisive sup.),
13=Ls (lèvre sup.), 14=Li (lèvre inf.), 15=Sn (subnasale),
16=Pog_s (pogonion tissu mou), 17=PNS, 18=ANS, 19=Ar

⚠️ Ce dataset ne contient AUCUN point d'apex/racine dentaire.
   Les mesures qui nécessitent l'axe complet de la dent (tip + apex) —
   SND, I/NA angle, i/NB angle, angle inter-incisif — ne peuvent donc
   PAS être calculées correctement et ne sont plus produites ici
   (mieux vaut les omettre que les fabriquer avec un point inventé).
"""

import numpy as np

# ── Ordre officiel ISBI 2015 ───────────────────────────────
LANDMARK_NAMES = [
    'S', 'Na', 'Or', 'Po', 'A', 'B', 'Pog', 'Me', 'Gn', 'Go',
    'L1', 'U1', 'Ls', 'Li', 'Sn', 'Pog_s', 'PNS', 'ANS', 'Ar'
]

# ── Normes de Steiner (uniquement les mesures calculables) ─
STEINER_NORMS = {
    'SNA'           : (82.0, 2.0),
    'SNB'           : (80.0, 2.0),
    'ANB'           : (2.0,  2.0),
    'Go_Gn_SN'      : (32.0, 5.0),
    'I_NA_distance' : (4.0,  1.0),
    'i_NB_distance' : (4.0,  1.0),
    'Occ_SN'        : (14.0, 2.0),
    'SE'            : (22.0, 2.0),
    'SL'            : (51.0, 2.0),
}


# ── Fonctions géométriques ─────────────────────────────────

def angle_between_lines(p1, p2, p3, p4):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p4) - np.array(p3)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def angle_point_to_line(point, line_p1, line_p2):
    v1 = np.array(point)   - np.array(line_p1)
    v2 = np.array(line_p2) - np.array(line_p1)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def distance_point_to_line(point, line_p1, line_p2):
    p0 = np.array(point,   dtype=float)
    p1 = np.array(line_p1, dtype=float)
    p2 = np.array(line_p2, dtype=float)
    d  = p2 - p1
    n  = np.array([-d[1], d[0]])
    n  = n / (np.linalg.norm(n) + 1e-8)
    return float(abs(np.dot(p0 - p1, n)))


def signed_distance_point_to_line(point, line_p1, line_p2):
    p0 = np.array(point,   dtype=float)
    p1 = np.array(line_p1, dtype=float)
    p2 = np.array(line_p2, dtype=float)
    d  = p2 - p1
    n  = np.array([-d[1], d[0]])
    n  = n / (np.linalg.norm(n) + 1e-8)
    return float(np.dot(p0 - p1, n))


def euclidean_distance(p1, p2):
    return float(np.linalg.norm(np.array(p2) - np.array(p1)))


def pixels_to_mm(pixels, scale=0.1):
    return pixels * scale


# ── Extraction des coordonnées ─────────────────────────────

def extract_coords(landmarks_array):
    """Convertit array (19,2) ou (38,) en dict {'S':(x,y), ...}"""
    if hasattr(landmarks_array, 'numpy'):
        landmarks_array = landmarks_array.numpy()
    landmarks_array = np.array(landmarks_array)
    if landmarks_array.ndim == 1:
        landmarks_array = landmarks_array.reshape(19, 2)
    return {name: (float(landmarks_array[i, 0]), float(landmarks_array[i, 1]))
            for i, name in enumerate(LANDMARK_NAMES)}


# ── Analyse de Steiner ─────────────────────────────────────

def compute_steiner_analysis(landmarks, img_w=1935, img_h=2400, scale=0.1):
    """
    Calcule les mesures de Steiner réellement calculables avec les
    19 points annotés du dataset ISBI 2015 (aucun point inventé).
    """
    if not isinstance(landmarks, dict):
        coords = extract_coords(landmarks)
        if all(0 <= v[0] <= 1 and 0 <= v[1] <= 1 for v in coords.values()):
            coords = {k: (v[0]*img_w, v[1]*img_h) for k, v in coords.items()}
    else:
        coords = landmarks

    # ── Points réellement annotés (aucun n'est fabriqué) ──
    S     = coords['S']
    Na    = coords['Na']
    A     = coords['A']
    B     = coords['B']
    Pog   = coords['Pog']
    Gn    = coords['Gn']
    Go    = coords['Go']
    L1    = coords['L1']      # pointe incisive inf.
    U1    = coords['U1']      # pointe incisive sup.
    Ls    = coords['Ls']      # lèvre sup.
    Li    = coords['Li']      # lèvre inf.
    Sn    = coords['Sn']      # subnasale
    Pog_s = coords['Pog_s']   # pogonion tissu mou

    results = {}

    # 1. Diagnostic squelettique — sens antéro-postérieur
    results['SNA'] = angle_point_to_line(A, Na, S)
    results['SNB'] = angle_point_to_line(B, Na, S)
    results['ANB'] = results['SNA'] - results['SNB']
    # SND retiré : nécessite le point D, absent de ce dataset

    # 2. Diagnostic squelettique — sens vertical
    results['Go_Gn_SN'] = angle_between_lines(S, Na, Go, Gn)
    results['SE'] = pixels_to_mm(euclidean_distance(S, Go), scale)
    results['SL'] = pixels_to_mm(euclidean_distance(S, Gn), scale)
    results['Ao_Bo'] = signed_distance_point_to_line(A, Na, B)

    # 3. Diagnostic dento-squelettique
    # (les angles I/NA et i/NB retirés : nécessitent l'apex/racine
    #  de l'incisive, absent de ce dataset — seule la distance,
    #  calculable avec la pointe incisive, est conservée)
    results['I_NA_distance'] = pixels_to_mm(distance_point_to_line(U1, Na, A), scale)
    results['i_NB_distance'] = pixels_to_mm(distance_point_to_line(L1, Na, B), scale)
    results['Occ_SN']        = angle_between_lines(S, Na, U1, L1)
    results['Pog_NB']        = pixels_to_mm(distance_point_to_line(Pog, Na, B), scale)

    # 4. Diagnostic dento-dentaire
    # angle inter-incisif retiré : nécessite les 2 apex (U1R, L1R),
    # absents de ce dataset

    # 5. Esthétique — Ligne S de Steiner (vraie formule, avec les
    # vrais points tissus mous disponibles dans ISBI 2015 : Sn et Pog_s)
    ligne_s_ls_px = signed_distance_point_to_line(Ls, Sn, Pog_s)
    ligne_s_li_px = signed_distance_point_to_line(Li, Sn, Pog_s)
    results['ligne_S_Ls'] = pixels_to_mm(abs(ligne_s_ls_px), scale)
    results['ligne_S_Li'] = pixels_to_mm(abs(ligne_s_li_px), scale)
    results['ligne_S_Ls_signe'] = 1.0 if ligne_s_ls_px > 0 else -1.0
    results['ligne_S_Li_signe'] = 1.0 if ligne_s_li_px > 0 else -1.0

    return results


# ── Formatage ──────────────────────────────────────────────

def format_results(results):
    formatted = []
    mesures_info = {
        'SNA':           ('Angle SNA',       '°',  '82° ± 2°', 'Prognathie maxillaire',    'Rétrognathie maxillaire', 'Maxillaire normal'),
        'SNB':           ('Angle SNB',       '°',  '80° ± 2°', 'Prognathie mandibulaire',  'Rétrognathie mandibulaire','Mandibule normale'),
        'ANB':           ('Angle ANB',       '°',  '2° ± 2°',  'Classe II squelettique',   'Classe III squelettique', 'Classe I squelettique'),
        'Go_Gn_SN':      ('Angle Go-Gn/SN',  '°',  '32° ± 5°', 'Hyperdivergence',          'Hypodivergence',          'Divergence normale'),
        'SE':            ('Distance SE',     'mm', '22mm ± 2', 'Corps mand. long',         'Corps mand. court',       'Longueur normale'),
        'SL':            ('Distance SL',     'mm', '51mm ± 2', 'Longueur mand. grande',    'Longueur mand. petite',   'Longueur normale'),
        'I_NA_distance': ('I/NA (distance)', 'mm', '4mm ± 1',  'Proalvéolie maxillaire',   'Rétrusion maxillaire',    'Position normale'),
        'i_NB_distance': ('i/NB (distance)', 'mm', '4mm ± 1',  'Proalvéolie mandibulaire', 'Rétrusion mandibulaire',  'Position normale'),
        'Occ_SN':        ('Plan occlusal/SN','°',  '14°',      'Plan occlusal incliné bas','Plan occlusal incliné haut','Plan occlusal normal'),
    }

    for key, (label, unite, norme, ih, ib, inorm) in mesures_info.items():
        if key not in results:
            continue
        valeur = results[key]
        norm_val, norm_tol = STEINER_NORMS.get(key, (valeur, 999))
        if valeur > norm_val + norm_tol:
            statut, interpretation = 'HIGH', ih
        elif valeur < norm_val - norm_tol:
            statut, interpretation = 'LOW', ib
        else:
            statut, interpretation = 'NORMAL', inorm
        formatted.append({
            'key': key, 'label': label, 'valeur': round(valeur, 1),
            'unite': unite, 'norme': norme, 'statut': statut,
            'interpretation': interpretation,
            'emoji': '✅' if statut == 'NORMAL' else '🔴'
        })
    return formatted


def print_steiner_report(results, formatted=None):
    if formatted is None:
        formatted = format_results(results)
    print('\n' + '═'*60)
    print('  ANALYSE CÉPHALOMÉTRIQUE DE STEINER')
    print('  Cabinet Dr. Redouane Chiguer')
    print('═'*60)
    for f in formatted:
        print(f"  {f['emoji']} {f['label']:25s} {f['valeur']:6.1f}{f['unite']:3s} "
              f"(norme: {f['norme']:12s}) → {f['interpretation']}")
    if 'ligne_S_Ls' in results:
        signe = 'en avant' if results['ligne_S_Ls_signe'] > 0 else 'en arrière'
        print(f"\n  💄 Lèvre sup. {signe} de la ligne S ({results['ligne_S_Ls']:.1f}mm)")
    if 'ligne_S_Li' in results:
        signe = 'en avant' if results['ligne_S_Li_signe'] > 0 else 'en arrière'
        print(f"  💄 Lèvre inf. {signe} de la ligne S ({results['ligne_S_Li']:.1f}mm)")
    print('═'*60)


# ── Test ───────────────────────────────────────────────────

if __name__ == '__main__':
    print('🧪 Test geometry.py — Ordre ISBI 2015 corrigé (sans point inventé)\n')

    test_landmarks = {
        'S'    : (968,  400),
        'Na'   : (1050, 350),
        'Or'   : (1200, 450),
        'Po'   : (800,  450),
        'A'    : (1100, 850),
        'B'    : (1050, 1100),
        'Pog'  : (1020, 1250),
        'Me'   : (980,  1380),
        'Gn'   : (1000, 1350),
        'Go'   : (780,  1200),
        'L1'   : (1100, 1000),
        'U1'   : (1120, 950),
        'Ls'   : (1130, 900),
        'Li'   : (1110, 980),
        'Sn'   : (1140, 820),
        'Pog_s': (1010, 1280),
        'PNS'  : (900,  800),
        'ANS'  : (1080, 800),
        'Ar'   : (750,  500),
    }

    results   = compute_steiner_analysis(test_landmarks)
    formatted = format_results(results)
    print_steiner_report(results, formatted)

    print('\n' + '='*45)
    print(f'  ✅ GEOMETRY OK — {len(results)} mesures calculées, aucun point inventé')
    print('='*45)