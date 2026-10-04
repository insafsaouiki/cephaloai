"""
CéphaloAI — heatmap_model.py
Insaf Saouiki — EMSI — Stage Cabinet Dr. Redouane Chiguer

Architecture "heatmap" (au lieu de régression directe de coordonnées).

Principe :
  Le modèle ne sort plus 38 nombres (19 x,y) mais 19 "cartes de
  chaleur" (une par landmark), de taille HEATMAP_SIZE x HEATMAP_SIZE.
  Sur chaque carte, les pixels proches de la position réelle du
  point sont proches de 1, le reste proche de 0 (gaussienne 2D
  centrée sur le point). La coordonnée finale est retrouvée en
  cherchant le pixel le plus "chaud" sur chaque carte, affiné par
  une moyenne pondérée locale (sous-pixel).

  C'est l'approche standard utilisée par la quasi-totalité des
  publications de pointe sur la détection de landmarks (visage,
  pose humaine, céphalométrie).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

N_LANDMARKS  = 19
IMG_SIZE     = 512
HEATMAP_SIZE = 128          # résolution des cartes de sortie
SIGMA        = 3.0          # écart-type de la gaussienne (en pixels, échelle heatmap)


# ══════════════════════════════════════════════════════════════
# 1. GÉNÉRATION DES CARTES CIBLES (utilisé par le dataset)
# ══════════════════════════════════════════════════════════════

def generate_heatmap(x_norm, y_norm, size=HEATMAP_SIZE, sigma=SIGMA):
    """
    Crée une gaussienne 2D centrée sur (x_norm, y_norm) — coordonnées
    normalisées [0,1] — sur une carte de taille (size, size).
    """
    x = x_norm * (size - 1)
    y = y_norm * (size - 1)
    xs = np.arange(size)
    ys = np.arange(size)
    xx, yy = np.meshgrid(xs, ys)
    heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


def generate_heatmaps_batch(landmarks_norm, size=HEATMAP_SIZE, sigma=SIGMA):
    """landmarks_norm : array (19, 2) normalisé [0,1] → retourne (19, size, size)"""
    maps = np.zeros((len(landmarks_norm), size, size), dtype=np.float32)
    for i, (x, y) in enumerate(landmarks_norm):
        maps[i] = generate_heatmap(x, y, size, sigma)
    return maps


# ══════════════════════════════════════════════════════════════
# 2. ARCHITECTURE
# ══════════════════════════════════════════════════════════════

class UpBlock(nn.Module):
    """Upsampling x2 + conv — évite les artefacts 'damier' des ConvTranspose2d."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class HeatmapCephaloNet(nn.Module):
    """
    EfficientNet-B4 (features spatiales, sans pooling)
        → [B, 1792, 16, 16]  (512/32 = 16)
    Décodeur (4 x UpBlock, x2 à chaque fois)
        16 → 32 → 64 → 128
    Conv finale 1x1 → 19 canaux (un par landmark)
    Sigmoid → valeurs [0,1] comparables aux gaussiennes cibles
    """

    def __init__(self, n_landmarks=N_LANDMARKS, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=pretrained,
            num_classes=0, global_pool=''
        )
        backbone_ch = self.backbone.num_features  # 1792

        self.decoder = nn.Sequential(
            UpBlock(backbone_ch, 256),   # 16 → 32
            UpBlock(256, 128),           # 32 → 64
            UpBlock(128, 64),            # 64 → 128
        )
        self.head = nn.Conv2d(64, n_landmarks, kernel_size=1)

    def forward(self, x):
        feat = self.backbone.forward_features(x)   # [B, 1792, 16, 16]
        feat = self.decoder(feat)                   # [B, 64, 128, 128]
        heatmaps = torch.sigmoid(self.head(feat))    # [B, 19, 128, 128]
        return heatmaps


# ══════════════════════════════════════════════════════════════
# 3. LOSS — MSE pondérée (compense le déséquilibre fond/point)
# ══════════════════════════════════════════════════════════════

class WeightedHeatmapLoss(nn.Module):
    """
    MSE classique, mais les pixels proches du centre du landmark
    comptent beaucoup plus que le fond (sinon le modèle apprend
    juste à prédire des cartes toutes noires, ce qui donne déjà
    une erreur très faible vu que 99% des pixels sont du fond).
    """
    def __init__(self, fg_weight=50.0):
        super().__init__()
        self.fg_weight = fg_weight

    def forward(self, pred, target):
        weight = 1.0 + target * self.fg_weight
        loss = weight * (pred - target) ** 2
        return loss.mean()


# ══════════════════════════════════════════════════════════════
# 4. EXTRACTION DES COORDONNÉES DEPUIS LES HEATMAPS
# ══════════════════════════════════════════════════════════════

def heatmaps_to_coords(heatmaps, refine_window=5):
    """
    heatmaps : tensor [B, 19, H, W] ou [19, H, W]
    Retourne : coordonnées normalisées [0,1], array (B, 19, 2) ou (19, 2)

    Méthode : argmax (position du pixel le plus chaud) + raffinement
    sous-pixel par moyenne pondérée locale autour de ce pixel.
    """
    single = (heatmaps.dim() == 3)
    if single:
        heatmaps = heatmaps.unsqueeze(0)

    B, N, H, W = heatmaps.shape
    hm = heatmaps.detach().cpu().numpy()
    coords = np.zeros((B, N, 2), dtype=np.float32)

    half = refine_window // 2
    for b in range(B):
        for n in range(N):
            m = hm[b, n]
            iy, ix = np.unravel_index(np.argmax(m), m.shape)

            y0, y1 = max(0, iy - half), min(H, iy + half + 1)
            x0, x1 = max(0, ix - half), min(W, ix + half + 1)
            patch = m[y0:y1, x0:x1]
            ys, xs = np.mgrid[y0:y1, x0:x1]
            total = patch.sum() + 1e-8
            refined_y = (patch * ys).sum() / total
            refined_x = (patch * xs).sum() / total

            coords[b, n, 0] = refined_x / (W - 1)
            coords[b, n, 1] = refined_y / (H - 1)

    return coords[0] if single else coords


def compute_mre_heatmap(pred_heatmaps, target_landmarks_norm, img_w=1935, img_h=2400, scale=0.1):
    """
    pred_heatmaps : [B, 19, H, W] (sortie du modèle)
    target_landmarks_norm : [B, 19, 2] coordonnées normalisées [0,1] (vérité terrain)
    Retourne : MRE moyen en mm, et array des erreurs par point
    """
    pred_coords = heatmaps_to_coords(pred_heatmaps)  # (B, 19, 2) normalisé
    target = target_landmarks_norm.cpu().numpy() if torch.is_tensor(target_landmarks_norm) \
        else target_landmarks_norm

    pred_px = pred_coords.copy()
    pred_px[..., 0] *= img_w
    pred_px[..., 1] *= img_h
    target_px = target.copy()
    target_px[..., 0] *= img_w
    target_px[..., 1] *= img_h

    dist_px = np.linalg.norm(pred_px - target_px, axis=-1)  # (B, 19)
    dist_mm = dist_px * scale
    return float(dist_mm.mean()), dist_mm
