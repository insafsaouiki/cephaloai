"""
CéphaloAI — model.py
Insaf Saouiki — EMSI — Stage Cabinet Dr. Redouane Chiguer

Ce module contient :
- L'architecture du modèle (EfficientNet-B4 + tête de régression)
- La Wing Loss (fonction de perte spécialisée pour les landmarks)
- Les métriques d'évaluation (MRE, SDR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

# ── Configuration ──────────────────────────────────────────
N_LANDMARKS = 19       # nombre de points à détecter
N_OUTPUTS   = N_LANDMARKS * 2   # 19 × (x, y) = 38 sorties
IMG_SIZE    = 512

LANDMARK_NAMES = [
    'S', 'Na', 'Or', 'Po', 'A', 'B', 'Pog', 'Me', 'Gn', 'Go',
    'L1', 'U1', 'Ls', 'Li', 'Sn', 'Pog_s', 'PNS', 'ANS', 'Ar'
]


# ══════════════════════════════════════════════════════════════
# 1. ARCHITECTURE DU MODÈLE
# ══════════════════════════════════════════════════════════════

class CephaloNet(nn.Module):
    """
    Modèle de détection de landmarks céphalométriques.

    Architecture :
        EfficientNet-B4 (backbone pré-entraîné ImageNet)
            ↓
        Global Average Pooling  (1792 features)
            ↓
        Fully Connected (1792 → 512, ReLU, Dropout 0.3)
            ↓
        Fully Connected (512 → 256, ReLU, Dropout 0.2)
            ↓
        Fully Connected (256 → 38)  ← 19 points × (x, y)
            ↓
        Sigmoid  → valeurs entre 0 et 1 (coordonnées normalisées)

    Pourquoi EfficientNet-B4 ?
    - Meilleur ratio précision / vitesse parmi les CNN modernes
    - Pré-entraîné sur ImageNet → transfer learning efficace
    - 19M paramètres → assez puissant sans être trop lourd
    """

    def __init__(self, n_outputs=N_OUTPUTS, pretrained=True, dropout=0.3):
        super(CephaloNet, self).__init__()

        # ── Backbone EfficientNet-B4 ───────────────────────
        # timm = librairie de modèles pré-entraînés
        # num_classes=0 → on enlève la tête de classification originale
        # on garde juste le feature extractor
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,          # pas de classification
            global_pool='avg'       # Global Average Pooling inclus
        )

        # Taille de sortie du backbone EfficientNet-B4
        backbone_out = self.backbone.num_features  # = 1792

        # ── Tête de régression ─────────────────────────────
        self.head = nn.Sequential(
            # Couche 1 : 1792 → 512
            nn.Linear(backbone_out, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            # Couche 2 : 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.7),

            # Couche finale : 256 → 38 (19 landmarks × x,y)
            nn.Linear(256, n_outputs),

            # Sigmoid → coordonnées entre 0 et 1
            nn.Sigmoid()
        )

        # Initialisation des poids de la tête
        self._init_weights()

    def _init_weights(self):
        """Initialisation He pour les couches linéaires"""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x : tensor (batch, 3, 512, 512)
        Returns:
            tensor (batch, 38) — coordonnées normalisées [0, 1]
        """
        features = self.backbone(x)   # (batch, 1792)
        output   = self.head(features) # (batch, 38)
        return output

    def freeze_backbone(self):
        """
        Gèle le backbone — phase 1 de l'entraînement.
        Seule la tête de régression est entraînée.
        Utile pour les premiers epochs (entraînement rapide).
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print('🔒 Backbone gelé — seule la tête est entraînée')

    def unfreeze_backbone(self, layers=None):
        """
        Dégèle le backbone — phase 2 (fine-tuning).
        Args:
            layers : None = dégeler tout le backbone
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print('🔓 Backbone dégelé — fine-tuning complet')

    def get_n_params(self):
        """Retourne le nombre de paramètres entraînables"""
        total  = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return total, trainable


# ══════════════════════════════════════════════════════════════
# 2. WING LOSS
# ══════════════════════════════════════════════════════════════

class WingLoss(nn.Module):
    """
    Wing Loss — fonction de perte spécialisée pour la détection
    de landmarks anatomiques.

    Référence : Wang et al., "Wing Loss for Robust Facial Landmark
    Localisation with Convolutional Neural Networks", CVPR 2018.

    Pourquoi Wing Loss et pas MSE ?
    ─────────────────────────────────────────────────────────────
    MSE pénalise pareil une erreur de 1px et une erreur de 20px.
    Ce n'est pas adapté aux landmarks médicaux où la précision
    au pixel près est critique.

    Wing Loss utilise :
    - Log pour les PETITES erreurs (< omega) → pénalité forte,
      force le modèle à être très précis sur les petits écarts
    - Linéaire pour les GRANDES erreurs → évite que les outliers
      dominent l'entraînement

    Formule :
        Si |x| < omega : omega * ln(1 + |x|/epsilon)
        Sinon          : |x| - C
        Où C = omega - omega * ln(1 + omega/epsilon)

    Paramètres recommandés :
        omega   = 10    (seuil entre zone log et zone linéaire)
        epsilon = 2     (régularisation de la zone log)
    """

    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega   = omega
        self.epsilon = epsilon
        # Constante C pour assurer la continuité de la fonction
        self.C = omega - omega * np.log(1 + omega / epsilon)

    def forward(self, pred, target):
        """
        Calcule la Wing Loss entre prédictions et cibles.

        Args:
            pred   : tensor (batch, 38) — coordonnées prédites
            target : tensor (batch, 38) — coordonnées réelles

        Returns:
            scalar — valeur moyenne de la loss
        """
        diff = torch.abs(pred - target)

        # Zone log (petites erreurs) : |x| < omega
        loss_log = self.omega * torch.log(
            1 + diff / self.epsilon
        )

        # Zone linéaire (grandes erreurs) : |x| >= omega
        loss_lin = diff - self.C

        # Appliquer la bonne formule selon la magnitude de l'erreur
        loss = torch.where(
            diff < self.omega,
            loss_log,
            loss_lin
        )

        return loss.mean()


# ══════════════════════════════════════════════════════════════
# 3. MÉTRIQUES D'ÉVALUATION
# ══════════════════════════════════════════════════════════════

def compute_mre(pred, target, img_w, img_h):
    """
    MRE — Mean Radial Error (en millimètres).

    C'est LA métrique de référence en céphalométrie.
    Standard clinique : MRE ≤ 2.0 mm = acceptable.

    Calcul :
    1. Dénormaliser les coordonnées [0,1] → pixels
    2. Calculer la distance euclidienne entre prédit et réel
    3. Convertir pixels → mm (résolution radio ~0.1 mm/pixel)
    4. Moyenner sur tous les landmarks et toutes les images

    Args:
        pred   : tensor (batch, 38) normalisé [0,1]
        target : tensor (batch, 38) normalisé [0,1]
        img_w  : largeur originale des images
        img_h  : hauteur originale des images

    Returns:
        mre_global : float — MRE moyen sur tous les landmarks (mm)
        mre_per_lm : array (19,) — MRE par landmark (mm)
    """
    # Résolution typique des radios céphalométriques ISBI 2015
    MM_PER_PIXEL = 0.1  # 1 pixel ≈ 0.1 mm

    pred_np   = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    batch_size = pred_np.shape[0]
    errors_per_lm = []

    for i in range(N_LANDMARKS):
        # Extraire x, y pour le landmark i
        pred_x   = pred_np[:, i*2]   * img_w
        pred_y   = pred_np[:, i*2+1] * img_h
        target_x = target_np[:, i*2]   * img_w
        target_y = target_np[:, i*2+1] * img_h

        # Distance euclidienne en pixels
        dist_px = np.sqrt(
            (pred_x - target_x)**2 + (pred_y - target_y)**2
        )

        # Convertir en mm
        dist_mm = dist_px * MM_PER_PIXEL
        errors_per_lm.append(dist_mm.mean())

    mre_per_lm = np.array(errors_per_lm)
    mre_global = mre_per_lm.mean()

    return mre_global, mre_per_lm


def compute_sdr(pred, target, img_w, img_h, thresholds=(2.0, 2.5, 3.0, 4.0)):
    """
    SDR — Successful Detection Rate.

    Pourcentage de landmarks détectés avec une erreur
    inférieure à un seuil donné (en mm).

    Standard papers 2024 :
        SDR @ 2.0mm ≥ 75% → bon modèle
        SDR @ 4.0mm ≥ 90% → bon modèle

    Args:
        thresholds : seuils en mm pour lesquels calculer le SDR

    Returns:
        dict {seuil: pourcentage}
    """
    MM_PER_PIXEL = 0.1

    pred_np   = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    all_distances = []

    for i in range(N_LANDMARKS):
        pred_x   = pred_np[:, i*2]   * img_w
        pred_y   = pred_np[:, i*2+1] * img_h
        target_x = target_np[:, i*2]   * img_w
        target_y = target_np[:, i*2+1] * img_h

        dist_mm = np.sqrt(
            (pred_x - target_x)**2 + (pred_y - target_y)**2
        ) * MM_PER_PIXEL

        all_distances.extend(dist_mm.tolist())

    all_distances = np.array(all_distances)

    sdr = {}
    for t in thresholds:
        sdr[t] = float((all_distances < t).mean() * 100)

    return sdr


def print_metrics(mre_global, mre_per_lm, sdr, epoch=None):
    """Affiche les métriques de manière lisible"""
    header = f'Epoch {epoch} — ' if epoch is not None else ''
    print(f'\n📊 {header}Métriques :')
    print(f'   MRE Global : {mre_global:.3f} mm', end='')

    if mre_global <= 2.0:
        print(' ✅ (objectif atteint)')
    elif mre_global <= 2.5:
        print(' ⚠️  (acceptable)')
    else:
        print(' ❌ (à améliorer)')

    print(f'\n   SDR :')
    for threshold, rate in sdr.items():
        status = '✅' if (threshold == 2.0 and rate >= 75) or \
                        (threshold == 4.0 and rate >= 90) else ''
        print(f'   @ {threshold}mm : {rate:.1f}% {status}')

    print(f'\n   MRE par landmark :')
    for i, name in enumerate(LANDMARK_NAMES):
        bar = '█' * int(mre_per_lm[i] / 0.2)
        status = '✅' if mre_per_lm[i] <= 2.0 else '⚠️ '
        print(f'   {name:5s} : {mre_per_lm[i]:.2f}mm {status} {bar}')


# ══════════════════════════════════════════════════════════════
# 4. TEST RAPIDE
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('🧪 Test model.py\n')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'💻 Device : {device}')
    print()

    # ── Test architecture ──────────────────────────────────
    print('1️⃣  Test architecture CephaloNet...')
    model = CephaloNet(pretrained=True)
    model = model.to(device)

    total, trainable = model.get_n_params()
    print(f'   ✅ Modèle créé')
    print(f'   Paramètres totaux      : {total:,}')
    print(f'   Paramètres entraînables: {trainable:,}')

    # ── Test forward pass ──────────────────────────────────
    print('\n2️⃣  Test forward pass...')
    dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f'   Input  : {tuple(dummy_input.shape)}')
    print(f'   Output : {tuple(output.shape)}')
    print(f'   Min/Max: {output.min():.3f} / {output.max():.3f}')
    print(f'   ✅ Toutes les valeurs entre 0 et 1 : '
          f'{(output >= 0).all() and (output <= 1).all()}')

    # ── Test Wing Loss ─────────────────────────────────────
    print('\n3️⃣  Test Wing Loss...')
    criterion = WingLoss(omega=10, epsilon=2)
    pred   = torch.rand(2, N_OUTPUTS).to(device)
    target = torch.rand(2, N_OUTPUTS).to(device)
    loss   = criterion(pred, target)
    print(f'   ✅ Wing Loss calculée : {loss.item():.4f}')

    # ── Test freeze/unfreeze ───────────────────────────────
    print('\n4️⃣  Test freeze/unfreeze...')
    model.freeze_backbone()
    _, trainable_frozen = model.get_n_params()
    print(f'   Params entraînables (gelé)  : {trainable_frozen:,}')

    model.unfreeze_backbone()
    _, trainable_full = model.get_n_params()
    print(f'   Params entraînables (complet): {trainable_full:,}')

    # ── Test métriques ─────────────────────────────────────
    print('\n5️⃣  Test métriques MRE / SDR...')
    pred_test   = torch.rand(4, N_OUTPUTS)
    target_test = torch.rand(4, N_OUTPUTS)
    mre_g, mre_lm = compute_mre(pred_test, target_test,
                                  img_w=1935, img_h=2400)
    sdr = compute_sdr(pred_test, target_test,
                      img_w=1935, img_h=2400)
    print_metrics(mre_g, mre_lm, sdr)

    print('\n' + '=' * 45)
    print('  ✅ MODEL OK — Prêt pour train.py !')
    print('=' * 45)
