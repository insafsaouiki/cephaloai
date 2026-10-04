"""
CéphaloAI — train.py
Insaf Saouiki — EMSI — Stage Cabinet Dr. Redouane Chiguer

Boucle d'entraînement complète :
- Phase 1 : backbone gelé (10 epochs) — entraîner la tête
- Phase 2 : fine-tuning complet (20 epochs) — affiner tout
- Sauvegarde du meilleur checkpoint
- Courbes de loss et MRE
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from tqdm import tqdm

# Importer nos modules
import sys
sys.path.append(str(Path(__file__).parent))
from preprocessing import get_dataloaders
from model import CephaloNet, WingLoss, compute_mre, compute_sdr, print_metrics

# ── Configuration ──────────────────────────────────────────
CONFIG = {
    # Données
    'img_size'      : 512,
    'batch_size'    : 8,       # réduit pour CPU (mettre 16 sur Colab GPU)
    'num_workers'   : 0,       # 0 sur Windows

    # Entraînement Phase 1 — backbone gelé
    'epochs_phase1' : 10,
    'lr_phase1'     : 1e-3,    # learning rate élevé pour la tête

    # Entraînement Phase 2 — fine-tuning complet
    'epochs_phase2' : 20,
    'lr_phase2'     : 1e-4,    # learning rate faible pour le backbone

    # Wing Loss
    'wing_omega'    : 10,
    'wing_epsilon'  : 2,

    # Scheduler
    'patience'      : 5,       # epochs sans amélioration avant réduction LR
    'lr_factor'     : 0.5,     # facteur de réduction du LR

    # Chemins
    'checkpoint_dir': 'models/checkpoints',
    'best_model'    : 'models/checkpoints/best_model.pth',
    'history_path'  : 'models/checkpoints/history.json',
    'plots_path'    : 'models/checkpoints/training_curves.png',

    # Dimensions originales ISBI 2015 (pour MRE en mm)
    'img_w'         : 1935,
    'img_h'         : 2400,
}


# ══════════════════════════════════════════════════════════════
# 1. FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, epoch, val_mre, config, is_best=False):
    """Sauvegarde le modèle"""
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch'     : epoch,
        'val_mre'   : val_mre,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'config'    : config,
    }
    # Sauvegarder le checkpoint courant
    path = Path(config['checkpoint_dir']) / f'checkpoint_epoch{epoch:03d}.pth'
    torch.save(checkpoint, path)

    # Sauvegarder le meilleur modèle séparément
    if is_best:
        torch.save(checkpoint, config['best_model'])
        print(f'   💾 Meilleur modèle sauvegardé ! (MRE={val_mre:.3f}mm)')


def plot_history(history, save_path):
    """Trace les courbes d'entraînement"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#1a1a2e')

    colors = {'train': '#4ECDC4', 'val': '#FF6B6B'}

    # Loss
    axes[0].plot(history['train_loss'], color=colors['train'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'],   color=colors['val'],   label='Val',   linewidth=2)
    axes[0].set_title('Wing Loss', color='white', fontweight='bold')
    axes[0].set_xlabel('Epoch', color='white')
    axes[0].set_ylabel('Loss', color='white')
    axes[0].legend()
    axes[0].set_facecolor('#2d2d44')
    axes[0].tick_params(colors='white')

    # MRE
    axes[1].plot(history['train_mre'], color=colors['train'], label='Train', linewidth=2)
    axes[1].plot(history['val_mre'],   color=colors['val'],   label='Val',   linewidth=2)
    axes[1].axhline(y=2.0, color='yellow', linestyle='--',
                    linewidth=1.5, label='Objectif 2mm')
    axes[1].set_title('MRE (mm)', color='white', fontweight='bold')
    axes[1].set_xlabel('Epoch', color='white')
    axes[1].set_ylabel('MRE (mm)', color='white')
    axes[1].legend()
    axes[1].set_facecolor('#2d2d44')
    axes[1].tick_params(colors='white')

    # Learning Rate
    axes[2].plot(history['lr'], color='#FFE66D', linewidth=2)
    axes[2].set_title('Learning Rate', color='white', fontweight='bold')
    axes[2].set_xlabel('Epoch', color='white')
    axes[2].set_ylabel('LR', color='white')
    axes[2].set_yscale('log')
    axes[2].set_facecolor('#2d2d44')
    axes[2].tick_params(colors='white')

    plt.suptitle('CéphaloAI — Courbes d\'entraînement',
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f'   📈 Courbes sauvegardées → {save_path}')


# ══════════════════════════════════════════════════════════════
# 2. BOUCLE D'ENTRAÎNEMENT — UNE EPOCH
# ══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, config):
    """
    Entraîne le modèle sur une epoch complète.
    Retourne : loss moyenne, MRE moyen
    """
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc='  Train', leave=False,
                bar_format='{l_bar}{bar:20}{r_bar}')

    for batch_imgs, batch_labels in pbar:
        batch_imgs   = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        # Forward
        optimizer.zero_grad()
        predictions = model(batch_imgs)

        # Calcul de la loss
        loss = criterion(predictions, batch_labels)

        # Backward
        loss.backward()

        # Gradient clipping — évite les explosions de gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        all_preds.append(predictions.detach().cpu())
        all_targets.append(batch_labels.detach().cpu())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Métriques sur toute l'epoch
    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mre, _      = compute_mre(all_preds, all_targets,
                               config['img_w'], config['img_h'])
    avg_loss    = total_loss / len(loader)
    return avg_loss, mre


# ══════════════════════════════════════════════════════════════
# 3. VALIDATION
# ══════════════════════════════════════════════════════════════

def validate(model, loader, criterion, device, config):
    """
    Évalue le modèle sur le set de validation.
    Retourne : loss, MRE global, MRE par landmark, SDR
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc='  Val  ', leave=False,
                    bar_format='{l_bar}{bar:20}{r_bar}')
        for batch_imgs, batch_labels in pbar:
            batch_imgs   = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            predictions = model(batch_imgs)
            loss        = criterion(predictions, batch_labels)

            total_loss += loss.item()
            all_preds.append(predictions.cpu())
            all_targets.append(batch_labels.cpu())

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mre_g, mre_lm = compute_mre(all_preds, all_targets,
                                  config['img_w'], config['img_h'])
    sdr           = compute_sdr(all_preds, all_targets,
                                 config['img_w'], config['img_h'])
    avg_loss      = total_loss / len(loader)

    return avg_loss, mre_g, mre_lm, sdr


# ══════════════════════════════════════════════════════════════
# 4. ENTRAÎNEMENT PRINCIPAL
# ══════════════════════════════════════════════════════════════

def train(config=CONFIG):
    """
    Entraînement complet en 2 phases :
    Phase 1 — backbone gelé (rapide, apprend les bases)
    Phase 2 — fine-tuning complet (lent, affine la précision)
    """
    print('🚀 Démarrage de l\'entraînement CéphaloAI\n')
    print('=' * 50)

    # ── Device ────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'💻 Device : {device}')
    if device.type == 'cuda':
        print(f'   GPU : {torch.cuda.get_device_name(0)}')
        print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    else:
        print('   ⚠️  CPU détecté — l\'entraînement sera lent')
        print('   💡 Conseil : utilise Google Colab pour le GPU gratuit')
    print()

    # ── DataLoaders ────────────────────────────────────────
    print('📂 Chargement des données...')
    loaders = get_dataloaders(batch_size=config['batch_size'],
                               num_workers=config['num_workers'])
    print()

    # ── Modèle ────────────────────────────────────────────
    print('🧠 Initialisation du modèle...')
    model     = CephaloNet(pretrained=True).to(device)
    criterion = WingLoss(omega=config['wing_omega'],
                         epsilon=config['wing_epsilon'])
    total, _ = model.get_n_params()
    print(f'   EfficientNet-B4 : {total:,} paramètres')
    print()

    # ── Historique ─────────────────────────────────────────
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mre' : [], 'val_mre' : [],
        'lr'        : []
    }
    best_mre = float('inf')
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════
    # PHASE 1 — Backbone gelé
    # ══════════════════════════════════════════════════════
    print('━' * 50)
    print('📌 PHASE 1 — Backbone gelé')
    print(f'   Epochs : {config["epochs_phase1"]}')
    print(f'   LR     : {config["lr_phase1"]}')
    print('━' * 50)

    model.freeze_backbone()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr_phase1'],
        weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                   factor=config['lr_factor'],
                                   patience=config['patience'])

    for epoch in range(1, config['epochs_phase1'] + 1):
        t0 = time.time()
        print(f'\n🔄 Epoch {epoch}/{config["epochs_phase1"]} (Phase 1)')

        train_loss, train_mre = train_one_epoch(
            model, loaders['train'], criterion, optimizer, device, config
        )
        val_loss, val_mre, val_mre_lm, val_sdr = validate(
            model, loaders['val'], criterion, device, config
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Sauvegarder historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mre'].append(train_mre)
        history['val_mre'].append(val_mre)
        history['lr'].append(current_lr)

        elapsed = time.time() - t0
        print(f'   Train Loss: {train_loss:.4f} | Train MRE: {train_mre:.2f}mm')
        print(f'   Val   Loss: {val_loss:.4f} | Val   MRE: {val_mre:.2f}mm')
        print(f'   LR: {current_lr:.2e} | Temps: {elapsed:.1f}s')

        # Sauvegarder si meilleur modèle
        is_best = val_mre < best_mre
        if is_best:
            best_mre = val_mre
        save_checkpoint(model, optimizer, epoch, val_mre, config, is_best)

    # ══════════════════════════════════════════════════════
    # PHASE 2 — Fine-tuning complet
    # ══════════════════════════════════════════════════════
    print(f'\n{"━"*50}')
    print('📌 PHASE 2 — Fine-tuning complet')
    print(f'   Epochs : {config["epochs_phase2"]}')
    print(f'   LR     : {config["lr_phase2"]}')
    print('━' * 50)

    model.unfreeze_backbone()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr_phase2'],
        weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                   factor=config['lr_factor'],
                                   patience=config['patience'])

    total_epochs = config['epochs_phase1'] + config['epochs_phase2']

    for epoch in range(config['epochs_phase1'] + 1, total_epochs + 1):
        t0 = time.time()
        phase2_ep = epoch - config['epochs_phase1']
        print(f'\n🔄 Epoch {epoch}/{total_epochs} (Phase 2 — {phase2_ep}/{config["epochs_phase2"]})')

        train_loss, train_mre = train_one_epoch(
            model, loaders['train'], criterion, optimizer, device, config
        )
        val_loss, val_mre, val_mre_lm, val_sdr = validate(
            model, loaders['val'], criterion, device, config
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mre'].append(train_mre)
        history['val_mre'].append(val_mre)
        history['lr'].append(current_lr)

        elapsed = time.time() - t0
        print(f'   Train Loss: {train_loss:.4f} | Train MRE: {train_mre:.2f}mm')
        print(f'   Val   Loss: {val_loss:.4f} | Val   MRE: {val_mre:.2f}mm')
        print(f'   LR: {current_lr:.2e} | Temps: {elapsed:.1f}s')

        is_best = val_mre < best_mre
        if is_best:
            best_mre = val_mre
        save_checkpoint(model, optimizer, epoch, val_mre, config, is_best)

        # Afficher métriques détaillées toutes les 5 epochs
        if epoch % 5 == 0:
            print_metrics(val_mre, val_mre_lm, val_sdr, epoch)

    # ══════════════════════════════════════════════════════
    # FIN — Résumé et visualisation
    # ══════════════════════════════════════════════════════
    print(f'\n{"="*50}')
    print('🏁 ENTRAÎNEMENT TERMINÉ !')
    print(f'   Meilleur MRE val : {best_mre:.3f} mm')
    print(f'   Modèle sauvegardé : {config["best_model"]}')

    # Sauvegarder l'historique
    with open(config['history_path'], 'w') as f:
        history_fixed = {k: [float(v) for v in vals] for k, vals in history.items()}
        json.dump(history_fixed, f, indent=2)
    print(f'   Historique sauvegardé : {config["history_path"]}')

    # Tracer les courbes
    plot_history(history, config['plots_path'])

    # Évaluation finale sur le test set
    print(f'\n{"─"*50}')
    print('📊 Évaluation finale sur le TEST SET...')
    checkpoint = torch.load(config['best_model'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    _, test_mre, test_mre_lm, test_sdr = validate(
        model, loaders['test'], criterion, device, config
    )
    print_metrics(test_mre, test_mre_lm, test_sdr)

    print(f'\n{"="*50}')
    if test_mre <= 2.0:
        print('  🎉 OBJECTIF ATTEINT — MRE ≤ 2.0mm !')
    elif test_mre <= 2.5:
        print('  ✅ BON RÉSULTAT — MRE acceptable')
    else:
        print('  ⚠️  À améliorer — continue le fine-tuning')
    print(f'{"="*50}')

    return model, history


# ══════════════════════════════════════════════════════════════
# 5. POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('╔══════════════════════════════════════════╗')
    print('║         CéphaloAI — Training             ║')
    print('║  Insaf Saouiki — EMSI — 2025             ║')
    print('╚══════════════════════════════════════════╝')
    print()
    print('⚠️  IMPORTANT :')
    print('   Ce script est optimisé pour Google Colab (GPU)')
    print('   Sur CPU local, 1 epoch ≈ 10-15 minutes')
    print('   Sur Colab GPU T4, 1 epoch ≈ 1-2 minutes')
    print()
    print('💡 Pour lancer sur Colab :')
    print('   1. Upload ce dossier sur Google Drive')
    print('   2. Ouvre training_colab.ipynb sur Colab')
    print('   3. Active le GPU : Runtime → Change runtime type → T4 GPU')
    print('   4. Lance toutes les cellules')
    print()

    reponse = input('Lancer l\'entraînement sur CPU local ? (o/n) : ')
    if reponse.lower() == 'o':
        model, history = train()
    else:
        print('👍 Lance le notebook Colab pour l\'entraînement GPU !')
        print('   Je vais générer training_colab.ipynb...')
