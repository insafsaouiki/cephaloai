"""
CéphaloAI — preprocessing.py
Insaf Saouiki — EMSI — Stage Cabinet Dr. Redouane Chiguer
"""

import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Configuration ──────────────────────────────────────────
IMG_SIZE    = 512
N_LANDMARKS = 19
DATA_DIR    = Path('data')
RAW_DIR     = DATA_DIR / 'raw'
IMG_DIR     = RAW_DIR / 'cepha400'
SPLITS_JSON = DATA_DIR / 'splits.json'

LANDMARK_NAMES = [
    'S', 'Na', 'Or', 'Po', 'A', 'B', 'Pog', 'Me', 'Gn', 'Go',
    'L1', 'U1', 'Ls', 'Li', 'Sn', 'Pog_s', 'PNS', 'ANS', 'Ar'
]


def preprocess_image(img_path, img_size=IMG_SIZE):
    """Lit une radio → CLAHE → resize → RGB → tensor normalisé ImageNet"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Image non trouvée : {img_path}')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


def normalize_landmarks(coords, orig_w, orig_h):
    """Coordonnées pixels → valeurs [0, 1]"""
    coords_norm = coords.copy().astype(np.float32)
    coords_norm[:, 0] /= orig_w
    coords_norm[:, 1] /= orig_h
    return coords_norm


def denormalize_landmarks(coords_norm, orig_w, orig_h):
    """Valeurs [0, 1] → coordonnées pixels"""
    coords = coords_norm.copy().astype(np.float32)
    coords[:, 0] *= orig_w
    coords[:, 1] *= orig_h
    return coords


def extract_landmarks(row):
    """Extrait les 19 landmarks d'une ligne CSV → array (19, 2)"""
    coords = []
    for i in range(1, 20):
        x = float(row[f'{i}_x'])
        y = float(row[f'{i}_y'])
        coords.append([x, y])
    return np.array(coords, dtype=np.float32)


class CephaloDataset(Dataset):
    """Dataset PyTorch — retourne image (3,512,512) + labels (38,)"""

    def __init__(self, df, img_dir, img_size=IMG_SIZE, augment=False):
        self.df       = df.reset_index(drop=True)
        self.img_dir  = Path(img_dir)
        self.img_size = img_size
        self.augment  = augment
        if augment:
            import albumentations as A
            self.aug_pipeline = A.Compose([
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.5),
                A.GaussNoise(p=0.3),
                A.RandomScale(scale_limit=0.1, p=0.3),
            ], keypoint_params=A.KeypointParams(format='xy',
                                                remove_invisible=False))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self.img_dir / row['image_path']
        img_orig = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_orig is None:
            raise FileNotFoundError(f'Image non trouvée : {img_path}')
        orig_h, orig_w = img_orig.shape
        landmarks = extract_landmarks(row)

        if self.augment:
            try:
                img_clahe   = cv2.createCLAHE(clipLimit=2.0,
                                               tileGridSize=(8,8)).apply(img_orig)
                img_resized = cv2.resize(img_clahe,
                                         (self.img_size, self.img_size))
                scale_x  = self.img_size / orig_w
                scale_y  = self.img_size / orig_h
                kps      = [(x*scale_x, y*scale_y) for x, y in landmarks]
                result   = self.aug_pipeline(image=img_resized, keypoints=kps)
                img_aug  = result['image']
                img_aug = cv2.resize(img_aug, (self.img_size, self.img_size))
                kps_aug  = result['keypoints']
                if len(kps_aug) == N_LANDMARKS:
                    lm_aug = np.array([[x/self.img_size, y/self.img_size]
                                       for x, y in kps_aug], dtype=np.float32)
                else:
                    lm_aug = normalize_landmarks(landmarks, orig_w, orig_h)
                img_rgb = cv2.cvtColor(img_aug, cv2.COLOR_GRAY2RGB)
                img_f   = img_rgb.astype(np.float32) / 255.0
                img_f   = (img_f - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
                return (torch.from_numpy(img_f.transpose(2,0,1)).float(),
                        torch.from_numpy(lm_aug.flatten()).float())
            except Exception:
                pass

        image_tensor   = preprocess_image(img_path, self.img_size)
        landmarks_norm = normalize_landmarks(landmarks, orig_w, orig_h)
        return image_tensor, torch.from_numpy(landmarks_norm.flatten()).float()


def get_dataloaders(splits_json=SPLITS_JSON, img_dir=IMG_DIR,
                    batch_size=16, num_workers=0):
    """Crée les DataLoaders train/val/test depuis splits.json"""
    with open(splits_json) as f:
        splits = json.load(f)

    # ✅ FIX : ajouter .jpg aux IDs du splits.json
    train_ids = [f"{s}.jpg" for s in splits['train']]
    val_ids   = [f"{s}.jpg" for s in splits['val']]
    test_ids  = [f"{s}.jpg" for s in splits['test']]

    dfs = []
    for csv_file in ['train_senior.csv', 'test1_senior.csv', 'test2_senior.csv']:
        csv_path = RAW_DIR / csv_file
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))
    df_all = pd.concat(dfs, ignore_index=True)

    df_train = df_all[df_all['image_path'].isin(train_ids)]
    df_val   = df_all[df_all['image_path'].isin(val_ids)]
    df_test  = df_all[df_all['image_path'].isin(test_ids)]

    print(f'📊 DataLoaders :')
    print(f'   Train : {len(df_train)} images')
    print(f'   Val   : {len(df_val)} images')
    print(f'   Test  : {len(df_test)} images')

    return {
        'train': DataLoader(CephaloDataset(df_train, img_dir, augment=True),
                            batch_size=batch_size, shuffle=True,
                            num_workers=num_workers),
        'val':   DataLoader(CephaloDataset(df_val, img_dir, augment=False),
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers),
        'test':  DataLoader(CephaloDataset(df_test, img_dir, augment=False),
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers),
    }


if __name__ == '__main__':
    print('🧪 Test preprocessing.py\n')

    test_img = IMG_DIR / '001.jpg'
    if test_img.exists():
        tensor = preprocess_image(test_img)
        print(f'✅ preprocess_image()')
        print(f'   Output : tensor {tuple(tensor.shape)}')
        print(f'   Min/Max: {tensor.min():.3f} / {tensor.max():.3f}')

    print()

    if SPLITS_JSON.exists():
        loaders = get_dataloaders(batch_size=4)
        print()
        batch_imgs, batch_labels = next(iter(loaders['train']))
        print(f'✅ Premier batch train :')
        print(f'   Images : {tuple(batch_imgs.shape)}')
        print(f'   Labels : {tuple(batch_labels.shape)}')
        print(f'   Labels min/max : {batch_labels.min():.3f} / {batch_labels.max():.3f}')
        print()
        print('=' * 45)
        print('  ✅ PREPROCESSING OK — Prêt pour model.py !')
        print('=' * 45)
    else:
        print(f'⚠️  splits.json non trouvé — lance exploration.ipynb d\'abord')
