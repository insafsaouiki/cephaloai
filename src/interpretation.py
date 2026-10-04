"""
CéphaloAI — interpretation.py
Insaf Saouiki — EMSI — Stage Cabinet Dr. Redouane Chiguer

Ce module interprète cliniquement les mesures de Steiner
et génère un diagnostic complet en langage orthodontique :
- Diagnostic squelettique (Classe I/II/III, hyper/hypodivergence)
- Diagnostic dento-squelettique (proalvéolie, rétrusion...)
- Diagnostic dento-dentaire (angle inter-incisif)
- Diagnostic esthétique (ligne S de Steiner)
- Signes structuraux de Bjork
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


# ══════════════════════════════════════════════════════════════
# 1. STRUCTURES DE DONNÉES
# ══════════════════════════════════════════════════════════════

@dataclass
class DiagnosticItem:
    """Un élément de diagnostic avec son niveau de sévérité."""
    categorie   : str    # 'squelettique', 'dento-squelettique', etc.
    mesure      : str    # 'SNA', 'ANB', etc.
    valeur      : float  # valeur calculée
    norme       : str    # norme de référence
    statut      : str    # 'NORMAL', 'LEGER', 'MODERE', 'SEVERE'
    diagnostic  : str    # interprétation clinique
    conseil     : str    # conseil thérapeutique


@dataclass
class DiagnosticComplet:
    """Diagnostic céphalométrique complet."""
    # Classifications principales
    classe_squelettique    : str   # 'Classe I', 'Classe II', 'Classe III'
    divergence             : str   # 'Normal', 'Hyperdivergent', 'Hypodivergent'
    maxillaire             : str   # 'Normal', 'Prognathique', 'Rétrognathique'
    mandibule              : str   # 'Normal', 'Prognathique', 'Rétrognathique'

    # Diagnostics détaillés
    squelettique           : List[DiagnosticItem]
    dento_squelettique     : List[DiagnosticItem]
    dento_dentaire         : List[DiagnosticItem]
    esthetique             : List[DiagnosticItem]

    # Résumé
    resume                 : str
    priorite_traitement    : str   # 'Urgente', 'Recommandée', 'Surveillance'
    emoji_global           : str   # '🔴', '🟠', '🟡', '✅'


# ══════════════════════════════════════════════════════════════
# 2. FONCTIONS D'INTERPRÉTATION
# ══════════════════════════════════════════════════════════════

def _get_statut(valeur, norme_val, norme_tol):
    """
    Détermine le statut clinique d'une mesure.
    
    Returns:
        'NORMAL'  → dans les limites
        'LEGER'   → écart < 2× la tolérance
        'MODERE'  → écart < 4× la tolérance
        'SEVERE'  → écart ≥ 4× la tolérance
    """
    ecart = abs(valeur - norme_val)
    if ecart <= norme_tol:
        return 'NORMAL'
    elif ecart <= norme_tol * 2:
        return 'LEGER'
    elif ecart <= norme_tol * 4:
        return 'MODERE'
    else:
        return 'SEVERE'


def interpreter_diagnostic_squelettique(results):
    """
    Interprète les mesures squelettiques de Steiner.
    Retourne liste de DiagnosticItem + classifications.
    """
    items = []
    classifications = {}

    # ── SNA ───────────────────────────────────────────────────
    if 'SNA' in results:
        sna = results['SNA']
        statut = _get_statut(sna, 82.0, 2.0)
        if sna > 84:
            diag    = f'Prognathie maxillaire (SNA={sna:.1f}°)'
            conseil = 'Recul maxillaire envisageable selon croissance'
            classif = 'Prognathique'
        elif sna < 80:
            diag    = f'Rétrognathie maxillaire (SNA={sna:.1f}°)'
            conseil = 'Avancée maxillaire ou compensation dentaire'
            classif = 'Rétrognathique'
        else:
            diag    = f'Maxillaire en position normale (SNA={sna:.1f}°)'
            conseil = 'Pas d\'intervention squelettique nécessaire'
            classif = 'Normal'
        classifications['maxillaire'] = classif
        items.append(DiagnosticItem(
            'squelettique', 'SNA', sna, '82° ± 2°',
            statut, diag, conseil
        ))

    # ── SNB ───────────────────────────────────────────────────
    if 'SNB' in results:
        snb = results['SNB']
        statut = _get_statut(snb, 80.0, 2.0)
        if snb > 82:
            diag    = f'Prognathie mandibulaire (SNB={snb:.1f}°)'
            conseil = 'Chirurgie de recul mandibulaire selon sévérité'
            classif = 'Prognathique'
        elif snb < 78:
            diag    = f'Rétrognathie mandibulaire (SNB={snb:.1f}°)'
            conseil = 'Avancée mandibulaire ou traitement fonctionnel'
            classif = 'Rétrognathique'
        else:
            diag    = f'Mandibule en position normale (SNB={snb:.1f}°)'
            conseil = 'Position mandibulaire satisfaisante'
            classif = 'Normal'
        classifications['mandibule'] = classif
        items.append(DiagnosticItem(
            'squelettique', 'SNB', snb, '80° ± 2°',
            statut, diag, conseil
        ))

    # ── ANB ───────────────────────────────────────────────────
    if 'ANB' in results:
        anb = results['ANB']
        statut = _get_statut(anb, 2.0, 2.0)
        if anb > 4:
            if anb <= 6:
                diag    = f'Classe II squelettique légère (ANB={anb:.1f}°)'
                conseil = 'Traitement orthodontique avec appareillage fonctionnel'
            elif anb <= 8:
                diag    = f'Classe II squelettique modérée (ANB={anb:.1f}°)'
                conseil = 'Traitement combiné orthodontie-chirurgie à envisager'
            else:
                diag    = f'Classe II squelettique sévère (ANB={anb:.1f}°)'
                conseil = 'Chirurgie orthognathique recommandée'
            classif = 'Classe II'
        elif anb < 0:
            diag    = f'Classe III squelettique (ANB={anb:.1f}°)'
            conseil = 'Traitement fonctionnel précoce ou chirurgie'
            classif = 'Classe III'
        else:
            diag    = f'Classe I squelettique (ANB={anb:.1f}°)'
            conseil = 'Relation maxillo-mandibulaire équilibrée'
            classif = 'Classe I'
        classifications['classe'] = classif
        items.append(DiagnosticItem(
            'squelettique', 'ANB', anb, '2° ± 2°',
            statut, diag, conseil
        ))

    # ── SND ───────────────────────────────────────────────────
    if 'SND' in results:
        snd = results['SND']
        statut = _get_statut(snd, 76.0, 2.0)
        if snd > 78:
            diag    = f'Implantation antérieure de la mandibule (SND={snd:.1f}°)'
            conseil = 'À corréler avec les autres mesures squelettiques'
        elif snd < 74:
            diag    = f'Implantation postérieure de la mandibule (SND={snd:.1f}°)'
            conseil = 'À corréler avec Go-Gn/SN pour diagnostic complet'
        else:
            diag    = f'Implantation mandibulaire normale (SND={snd:.1f}°)'
            conseil = 'Position symphysaire satisfaisante'
        items.append(DiagnosticItem(
            'squelettique', 'SND', snd, '76°',
            statut, diag, conseil
        ))

    # ── Go-Gn/SN ──────────────────────────────────────────────
    if 'Go_Gn_SN' in results:
        gogn = results['Go_Gn_SN']
        statut = _get_statut(gogn, 32.0, 5.0)
        if gogn > 37:
            diag    = f'Hyperdivergence — face longue (Go-Gn/SN={gogn:.1f}°)'
            conseil = 'Contrôle vertical strict, éviter les extractions'
            classif = 'Hyperdivergent'
        elif gogn < 27:
            diag    = f'Hypodivergence — face courte (Go-Gn/SN={gogn:.1f}°)'
            conseil = 'Extractions possibles, contrôle de la dimension verticale'
            classif = 'Hypodivergent'
        else:
            diag    = f'Divergence faciale normale (Go-Gn/SN={gogn:.1f}°)'
            conseil = 'Dimension verticale équilibrée'
            classif = 'Normal'
        classifications['divergence'] = classif
        items.append(DiagnosticItem(
            'squelettique', 'Go_Gn_SN', gogn, '32° ± 5°',
            statut, diag, conseil
        ))

    # ── SE et SL ──────────────────────────────────────────────
    if 'SE' in results:
        se = results['SE']
        statut = _get_statut(se, 22.0, 2.0)
        diag = f'Longueur mandibulaire postérieure : {se:.1f}mm'
        conseil = 'Longueur corps mandibulaire à surveiller en croissance'
        items.append(DiagnosticItem(
            'squelettique', 'SE', se, '22mm ± 2',
            statut, diag, conseil
        ))

    if 'SL' in results:
        sl = results['SL']
        statut = _get_statut(sl, 51.0, 2.0)
        diag = f'Longueur mandibulaire totale : {sl:.1f}mm'
        conseil = 'Longueur totale à corréler avec SE'
        items.append(DiagnosticItem(
            'squelettique', 'SL', sl, '51mm ± 2',
            statut, diag, conseil
        ))

    return items, classifications


def interpreter_dento_squelettique(results):
    """Interprète les mesures dento-squelettiques."""
    items = []

    # ── I/NA angle ────────────────────────────────────────────
    if 'I_NA_angle' in results:
        val    = results['I_NA_angle']
        statut = _get_statut(val, 22.0, 2.0)
        if val > 24:
            diag    = f'Vestibulo-version incisive sup (I/NA={val:.1f}°)'
            conseil = 'Ingression et linguoversion des incisives sup'
        elif val < 20:
            diag    = f'Linguo-version incisive sup (I/NA={val:.1f}°)'
            conseil = 'Protrusion des incisives sup envisageable'
        else:
            diag    = f'Inclinaison incisive sup normale (I/NA={val:.1f}°)'
            conseil = 'Pas de correction inclinale nécessaire'
        items.append(DiagnosticItem(
            'dento-squelettique', 'I/NA angle', val,
            '22° ± 2°', statut, diag, conseil
        ))

    # ── I/NA distance ─────────────────────────────────────────
    if 'I_NA_distance' in results:
        val    = results['I_NA_distance']
        statut = _get_statut(val, 4.0, 1.0)
        if val > 5:
            diag    = f'Proalvéolie maxillaire (I/NA={val:.1f}mm)'
            conseil = 'Recul des incisives sup indiqué'
        elif val < 3:
            diag    = f'Rétrusion maxillaire (I/NA={val:.1f}mm)'
            conseil = 'Avancée des incisives sup possible'
        else:
            diag    = f'Position incisive sup normale (I/NA={val:.1f}mm)'
            conseil = 'Position satisfaisante'
        items.append(DiagnosticItem(
            'dento-squelettique', 'I/NA distance', val,
            '4mm ± 1', statut, diag, conseil
        ))

    # ── i/NB angle ────────────────────────────────────────────
    if 'i_NB_angle' in results:
        val    = results['i_NB_angle']
        statut = _get_statut(val, 25.0, 2.0)
        if val > 27:
            diag    = f'Vestibulo-version incisive inf (i/NB={val:.1f}°)'
            conseil = 'Linguoversion des incisives inf'
        elif val < 23:
            diag    = f'Linguo-version incisive inf (i/NB={val:.1f}°)'
            conseil = 'Vestibulo-version des incisives inf'
        else:
            diag    = f'Inclinaison incisive inf normale (i/NB={val:.1f}°)'
            conseil = 'Inclinaison satisfaisante'
        items.append(DiagnosticItem(
            'dento-squelettique', 'i/NB angle', val,
            '25° ± 2°', statut, diag, conseil
        ))

    # ── i/NB distance ─────────────────────────────────────────
    if 'i_NB_distance' in results:
        val    = results['i_NB_distance']
        statut = _get_statut(val, 4.0, 1.0)
        if val > 5:
            diag    = f'Proalvéolie mandibulaire (i/NB={val:.1f}mm)'
            conseil = 'Recul des incisives inf indiqué'
        elif val < 3:
            diag    = f'Rétrusion mandibulaire (i/NB={val:.1f}mm)'
            conseil = 'Avancée des incisives inf possible'
        else:
            diag    = f'Position incisive inf normale (i/NB={val:.1f}mm)'
            conseil = 'Position satisfaisante'
        items.append(DiagnosticItem(
            'dento-squelettique', 'i/NB distance', val,
            '4mm ± 1', statut, diag, conseil
        ))

    # ── Biproalvéolie ─────────────────────────────────────────
    if ('I_NA_distance' in results and 'i_NB_distance' in results):
        if results['I_NA_distance'] > 5 and results['i_NB_distance'] > 5:
            items.append(DiagnosticItem(
                'dento-squelettique', 'Biproalvéolie', 0,
                '—', 'MODERE',
                'Biproalvéolie maxillo-mandibulaire',
                'Extractions et recul des incisives dans les deux arcades'
            ))

    # ── Occ/SN ────────────────────────────────────────────────
    if 'Occ_SN' in results:
        val    = results['Occ_SN']
        statut = _get_statut(val, 14.0, 2.0)
        if val > 16:
            diag    = f'Plan occlusal incliné vers le bas (Occ/SN={val:.1f}°)'
            conseil = 'Contrôle du plan occlusal lors du traitement'
        elif val < 12:
            diag    = f'Plan occlusal incliné vers le haut (Occ/SN={val:.1f}°)'
            conseil = 'Plan occlusal à surveiller'
        else:
            diag    = f'Plan occlusal normal (Occ/SN={val:.1f}°)'
            conseil = 'Plan occlusal satisfaisant'
        items.append(DiagnosticItem(
            'dento-squelettique', 'Occ/SN', val,
            '14°', statut, diag, conseil
        ))

    # ── Pog/NB ────────────────────────────────────────────────
    if 'Pog_NB' in results:
        val  = results['Pog_NB']
        diag = f'Distance Pogonion-NB : {val:.1f}mm'
        if 'i_NB_distance' in results:
            diff = abs(val - results['i_NB_distance'])
            if diff < 1:
                conseil = 'Équilibre dento-squelettique de Holdaway respecté'
            else:
                conseil = f'Écart Pog/NB vs i/NB = {diff:.1f}mm — déséquilibre'
        else:
            conseil = 'À comparer avec i/NB pour règle de Holdaway'
        items.append(DiagnosticItem(
            'dento-squelettique', 'Pog/NB', val,
            '= i/NB', 'NORMAL', diag, conseil
        ))

    return items


def interpreter_dento_dentaire(results):
    """Interprète les mesures dento-dentaires."""
    items = []

    if 'inter_incisif' in results:
        val    = results['inter_incisif']
        statut = _get_statut(val, 131.0, 5.0)
        if val < 126:
            diag    = f'Biproalvéolie (angle inter-incisif={val:.1f}°)'
            conseil = 'Recul des incisives dans les deux arcades'
        elif val > 136:
            diag    = f'Birétrolvéolie (angle inter-incisif={val:.1f}°)'
            conseil = 'Avancée des incisives dans les deux arcades'
        else:
            diag    = f'Relation inter-incisive normale ({val:.1f}°)'
            conseil = 'Angle inter-incisif satisfaisant'
        items.append(DiagnosticItem(
            'dento-dentaire', 'Angle inter-incisif', val,
            '131° ± 5°', statut, diag, conseil
        ))

    return items


def interpreter_esthetique(results):
    """Interprète le diagnostic esthétique (ligne S de Steiner)."""
    items = []

    if 'ligne_S_Ls' in results:
        val   = results['ligne_S_Ls']
        signe = results.get('ligne_S_Ls_signe', 1)

        if signe > 0:
            direction = 'en avant'
            diag = f'Prochélie — lèvres en avant de la ligne S ({val:.1f}mm)'
            if val > 3:
                conseil = 'Recul des lèvres souhaitable — compensation dentaire'
                statut  = 'MODERE'
            else:
                conseil = 'Légère prochélie — surveiller'
                statut  = 'LEGER'
        else:
            direction = 'en arrière'
            diag = f'Rétrocheilie — lèvres en arrière de la ligne S ({val:.1f}mm)'
            if val > 3:
                conseil = 'Avancée des lèvres souhaitable'
                statut  = 'MODERE'
            else:
                conseil = 'Légère rétrocheilie — surveiller'
                statut  = 'LEGER'

        if val < 1:
            diag    = 'Profil esthétique équilibré — lèvres sur la ligne S'
            conseil = 'Esthétique labiale satisfaisante'
            statut  = 'NORMAL'

        items.append(DiagnosticItem(
            'esthétique', 'Ligne S Steiner', val,
            'Lèvres tangentes', statut, diag, conseil
        ))

    return items


def interpreter_bjork(results):
    """
    Interprète les signes structuraux de Bjork.
    Détermine le type de rotation mandibulaire.
    Basé sur Go-Gn/SN et ANB principalement.
    """
    bjork = {}

    if 'Go_Gn_SN' in results and 'ANB' in results:
        gogn = results['Go_Gn_SN']
        anb  = results['ANB']

        if gogn < 27 and anb > 0:
            bjork['type'] = 'Rotation antérieure'
            bjork['description'] = (
                'Tendance à la rotation antérieure mandibulaire : '
                'angle intermolaire ouvert, hauteur étage inférieur diminuée'
            )
            bjork['pronostic'] = 'Favorable pour le traitement orthodontique'
        elif gogn > 37:
            bjork['type'] = 'Rotation postérieure'
            bjork['description'] = (
                'Tendance à la rotation postérieure mandibulaire : '
                'angle intermolaire fermé, hauteur étage inférieur augmentée'
            )
            bjork['pronostic'] = 'Traitement plus complexe — contrôle vertical strict'
        else:
            bjork['type'] = 'Rotation neutre'
            bjork['description'] = 'Rotation mandibulaire dans les limites normales'
            bjork['pronostic'] = 'Pronostic favorable'

    return bjork


# ══════════════════════════════════════════════════════════════
# 3. DIAGNOSTIC COMPLET
# ══════════════════════════════════════════════════════════════

def generer_diagnostic_complet(results):
    """
    Génère le diagnostic céphalométrique complet.

    Args:
        results : dict des mesures de Steiner (sortie de geometry.py)

    Returns:
        DiagnosticComplet avec toutes les interprétations
    """
    # Interpréter chaque section
    items_squelettique, classif = interpreter_diagnostic_squelettique(results)
    items_dento_squelettique    = interpreter_dento_squelettique(results)
    items_dento_dentaire        = interpreter_dento_dentaire(results)
    items_esthetique            = interpreter_esthetique(results)
    bjork                       = interpreter_bjork(results)

    # Classifications principales
    classe     = classif.get('classe',      'Classe I')
    divergence = classif.get('divergence',  'Normal')
    maxillaire = classif.get('maxillaire',  'Normal')
    mandibule  = classif.get('mandibule',   'Normal')

    # Résumé clinique
    resume_parts = []
    if classe != 'Classe I':
        resume_parts.append(classe)
    if divergence != 'Normal':
        resume_parts.append(divergence)
    if maxillaire != 'Normal':
        resume_parts.append(f'Maxillaire {maxillaire.lower()}')
    if mandibule != 'Normal':
        resume_parts.append(f'Mandibule {mandibule.lower()}')

    if resume_parts:
        resume = ' — '.join(resume_parts)
    else:
        resume = 'Occlusion équilibrée — pas d\'anomalie majeure détectée'

    # Ajouter Bjork au résumé
    if bjork.get('type') and bjork['type'] != 'Rotation neutre':
        resume += f' — {bjork["type"]}'

    # Priorité de traitement
    all_items = (items_squelettique + items_dento_squelettique +
                 items_dento_dentaire + items_esthetique)
    severe_count = sum(1 for i in all_items if i.statut == 'SEVERE')
    modere_count = sum(1 for i in all_items if i.statut == 'MODERE')

    if severe_count >= 2:
        priorite = 'Urgente'
        emoji    = '🔴'
    elif severe_count >= 1 or modere_count >= 2:
        priorite = 'Recommandée'
        emoji    = '🟠'
    elif modere_count >= 1:
        priorite = 'Surveillance'
        emoji    = '🟡'
    else:
        priorite = 'Contrôle annuel'
        emoji    = '✅'

    return DiagnosticComplet(
        classe_squelettique  = classe,
        divergence           = divergence,
        maxillaire           = maxillaire,
        mandibule            = mandibule,
        squelettique         = items_squelettique,
        dento_squelettique   = items_dento_squelettique,
        dento_dentaire       = items_dento_dentaire,
        esthetique           = items_esthetique,
        resume               = resume,
        priorite_traitement  = priorite,
        emoji_global         = emoji,
    )


# ══════════════════════════════════════════════════════════════
# 4. AFFICHAGE DU RAPPORT
# ══════════════════════════════════════════════════════════════

def afficher_diagnostic(diagnostic, nom_patient='[Patient]'):
    """Affiche le diagnostic complet dans le terminal."""

    STATUT_EMOJI = {
        'NORMAL' : '✅',
        'LEGER'  : '🟡',
        'MODERE' : '🟠',
        'SEVERE' : '🔴',
    }

    print('\n' + '╔' + '═'*58 + '╗')
    print('║' + '  RAPPORT DE DIAGNOSTIC CÉPHALOMÉTRIQUE'.center(58) + '║')
    print('║' + f'  Patient : {nom_patient}'.ljust(58) + '║')
    print('║' + '  Cabinet Dr. Redouane Chiguer — CéphaloAI'.ljust(58) + '║')
    print('╠' + '═'*58 + '╣')

    # Résumé global
    print(f'║  {diagnostic.emoji_global} DIAGNOSTIC : {diagnostic.resume[:50]}'.ljust(59) + '║')
    print(f'║  🎯 Classe squelettique : {diagnostic.classe_squelettique}'.ljust(59) + '║')
    print(f'║  📐 Divergence          : {diagnostic.divergence}'.ljust(59) + '║')
    print(f'║  ⚕️  Priorité traitement  : {diagnostic.priorite_traitement}'.ljust(59) + '║')
    print('╠' + '═'*58 + '╣')

    sections = [
        ('📋 DIAGNOSTIC SQUELETTIQUE',      diagnostic.squelettique),
        ('🦷 DIAGNOSTIC DENTO-SQUELETTIQUE', diagnostic.dento_squelettique),
        ('🔗 DIAGNOSTIC DENTO-DENTAIRE',     diagnostic.dento_dentaire),
        ('💄 DIAGNOSTIC ESTHÉTIQUE',         diagnostic.esthetique),
    ]

    for titre, items in sections:
        if not items:
            continue
        print(f'║  {titre}'.ljust(59) + '║')
        print('║  ' + '─'*55 + ' ║')
        for item in items:
            emoji  = STATUT_EMOJI.get(item.statut, 'ℹ️')
            ligne1 = f'  {emoji} {item.diagnostic}'[:56]
            ligne2 = f'     → {item.conseil}'[:56]
            print(f'║{ligne1.ljust(58)}║')
            print(f'║{ligne2.ljust(58)}║')
        print('║' + ' '*58 + '║')

    print('╠' + '═'*58 + '╣')
    print('║  ⚠️  Ce rapport est une aide au diagnostic,'.ljust(59) + '║')
    print('║     non un diagnostic médical définitif.'.ljust(59) + '║')
    print('║     Validation clinique indispensable.'.ljust(59) + '║')
    print('╚' + '═'*58 + '╝')


def diagnostic_to_dict(diagnostic):
    """Convertit le diagnostic en dictionnaire pour Streamlit/PDF."""
    def items_to_list(items):
        return [
            {
                'mesure'        : i.mesure,
                'valeur'        : i.valeur,
                'norme'         : i.norme,
                'statut'        : i.statut,
                'diagnostic'    : i.diagnostic,
                'conseil'       : i.conseil,
            }
            for i in items
        ]

    return {
        'classe_squelettique' : diagnostic.classe_squelettique,
        'divergence'          : diagnostic.divergence,
        'maxillaire'          : diagnostic.maxillaire,
        'mandibule'           : diagnostic.mandibule,
        'resume'              : diagnostic.resume,
        'priorite'            : diagnostic.priorite_traitement,
        'emoji'               : diagnostic.emoji_global,
        'squelettique'        : items_to_list(diagnostic.squelettique),
        'dento_squelettique'  : items_to_list(diagnostic.dento_squelettique),
        'dento_dentaire'      : items_to_list(diagnostic.dento_dentaire),
        'esthetique'          : items_to_list(diagnostic.esthetique),
    }


# ══════════════════════════════════════════════════════════════
# 5. TEST RAPIDE
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('🧪 Test interpretation.py\n')

    # Simuler des mesures de Steiner typiques — Classe II
    results_test = {
        'SNA'           : 85.0,   # Prognathie max légère
        'SNB'           : 78.0,   # Rétrognathie mand légère
        'ANB'           : 7.0,    # Classe II modérée
        'SND'           : 74.0,   # Implantation post
        'Go_Gn_SN'      : 38.0,   # Hyperdivergence légère
        'SE'            : 20.0,   # Légèrement court
        'SL'            : 48.0,   # Légèrement court
        'I_NA_angle'    : 26.0,   # Vestibulo-version sup
        'I_NA_distance' : 6.0,    # Proalvéolie max
        'i_NB_angle'    : 28.0,   # Vestibulo-version inf
        'i_NB_distance' : 5.5,    # Proalvéolie mand
        'Occ_SN'        : 17.0,   # Plan incliné
        'Pog_NB'        : 4.0,
        'inter_incisif' : 120.0,  # Biproalvéolie
        'ligne_S'       : 4.0,
        'ligne_S_signe' : 1.0,    # Prochélie
    }

    print('📊 Mesures de test (cas Classe II avec biproalvéolie) :')
    for k, v in list(results_test.items())[:5]:
        print(f'   {k}: {v}')
    print('   ...\n')

    # Générer le diagnostic
    diagnostic = generer_diagnostic_complet(results_test)

    # Afficher
    afficher_diagnostic(diagnostic, nom_patient='Patient Test')

    # Vérifications
    print(f'\n✅ Classe détectée     : {diagnostic.classe_squelettique}')
    print(f'✅ Divergence          : {diagnostic.divergence}')
    print(f'✅ Priorité traitement : {diagnostic.priorite_traitement}')
    print(f'✅ Emoji global        : {diagnostic.emoji_global}')

    # Test conversion dict
    d = diagnostic_to_dict(diagnostic)
    print(f'✅ Conversion dict     : {len(d)} clés')

    print('\n' + '='*45)
    print('  ✅ INTERPRETATION OK — Prêt pour app.py !')
    print('='*45)
