"""
TCMP-300 Combined Pipeline
Phase 1 : Swin Transformer training  (plots S1-S5)
Phase 2 : ML classifiers + ensemble  (plots S6, M1-M12)
"""

# =============================================================================
# IMPORTS
# =============================================================================

import json, time, random, warnings, threading, contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, datasets
import timm

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from xgboost import XGBClassifier

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def _make_autocast():
    if torch.cuda.is_available():
        return lambda: torch.amp.autocast('cuda')
    return contextlib.nullcontext


def _make_scaler():
    if torch.cuda.is_available():
        return torch.amp.GradScaler('cuda')
    return None


def safe_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except Exception:
        return Image.new('RGB', (224, 224), (128, 128, 128))


def savefig(path, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {path}")


def make_tsne(n_components=2, perplexity=40, n_iter=1000, random_state=42):
    try:
        return TSNE(n_components=n_components, perplexity=perplexity,
                    max_iter=n_iter, random_state=random_state)
    except TypeError:
        return TSNE(n_components=n_components, perplexity=perplexity,
                    n_iter=n_iter, random_state=random_state)


def train_with_timeout(clf, X_train, y_train, timeout_sec=None):
    result = {'done': False, 'error': None}

    def _worker():
        try:
            clf.fit(X_train, y_train)
            result['done'] = True
        except Exception as e:
            result['error'] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    return result['done'], result.get('error')


# =============================================================================
# DATA TRANSFORMS
# =============================================================================

class BalancedAug:
    """Conservative augmentation to keep train-val gap below 2%."""

    @staticmethod
    def train(img_size):
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.82, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                   saturation=0.1, hue=0.04),
            transforms.RandomGrayscale(p=0.03),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def val(img_size):
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


class HybridMix:
    """Mixup and CutMix applied with low probability to control the train-val gap."""

    def __init__(self, prob=0.12, mixup_alpha=0.6, cutmix_alpha=0.8):
        self.prob = prob
        self.ma   = mixup_alpha
        self.ca   = cutmix_alpha

    def mixup(self, x, y):
        lam = np.random.beta(self.ma, self.ma)
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam

    def cutmix(self, x, y):
        lam = np.random.beta(self.ca, self.ca)
        idx = torch.randperm(x.size(0), device=x.device)
        _, _, H, W = x.shape
        r  = np.sqrt(1 - lam)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = int(np.clip(cx - W * r / 2, 0, W))
        y1 = int(np.clip(cy - H * r / 2, 0, H))
        x2 = int(np.clip(cx + W * r / 2, 0, W))
        y2 = int(np.clip(cy + H * r / 2, 0, H))
        x  = x.clone()
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        return x, y, y[idx], lam

    def __call__(self, x, y):
        if random.random() < self.prob:
            fn = self.mixup if random.random() < 0.5 else self.cutmix
            return fn(x, y)
        return x, y, y, 1.0


class TTADataset(Dataset):
    """Produces 10 views per image (5-crop x 2 flips) for test-time augmentation."""

    def __init__(self, base_dataset, img_size=224):
        self.base     = base_dataset
        self.img_size = img_size
        resize        = int(img_size * 1.14)
        mean          = [0.485, 0.456, 0.406]
        std           = [0.229, 0.224, 0.225]
        self.resize_t  = transforms.Resize(resize)
        self.five_crop = transforms.FiveCrop(img_size)
        self.hflip     = transforms.RandomHorizontalFlip(p=1.0)
        self.post      = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        try:
            with open(path, 'rb') as f:
                pil = Image.open(f).convert('RGB')
        except Exception:
            pil = Image.new('RGB', (self.img_size, self.img_size), (128, 128, 128))
        pil   = self.resize_t(pil)
        crops = list(self.five_crop(pil))
        views = []
        for c in crops:
            views.append(self.post(c))
            views.append(self.post(self.hflip(c)))
        return torch.stack(views, dim=0), label


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class SwinClassifier(nn.Module):
    """Swin Transformer backbone with a custom two-layer classification head."""

    def __init__(self, num_classes: int, dropout: float = 0.25, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224.ms_in22k',
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=0.15,
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def get_features(self, x):
        return self.backbone(x)


class MultiScaleExtractor:
    """
    Hooks into an intermediate Swin stage to produce multi-scale features.
    Concatenates stage-2 pooled features with the final backbone output.
    """

    def __init__(self, model):
        self.model   = model
        self._buf    = []
        self._handle = model.backbone.layers[1].register_forward_hook(
            lambda m, inp, out: self._buf.append(out.detach().cpu()))
        self.pool = nn.AdaptiveAvgPool2d(1)

    @torch.no_grad()
    def extract(self, loader, device, autocast_ctx, desc='  Extracting features'):
        self.model.eval()
        all_feats, all_labels = [], []
        for imgs, lbls in tqdm(loader, desc=desc, ncols=90, leave=False):
            self._buf.clear()
            imgs = imgs.to(device, non_blocking=True)
            with autocast_ctx():
                f3 = self.model.backbone(imgs)
            f2_raw = self._buf[0].to(f3.device)
            f2     = self.pool(f2_raw).flatten(start_dim=1)
            combined = torch.cat([f2, f3], dim=1)
            all_feats.append(combined.cpu().numpy())
            all_labels.append(lbls.numpy())
        return np.vstack(all_feats), np.concatenate(all_labels)

    def remove(self):
        self._handle.remove()


# =============================================================================
# OPTIMIZER & SCHEDULER
# =============================================================================

def make_param_groups(model, peak_lr, bbone_mult=0.08, weight_decay=0.02):
    """Separate parameter groups for backbone (lower LR) and head (full LR)."""
    bb, hd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb if 'backbone' in n else hd).append(p)
    return [
        {'params': bb, 'lr': peak_lr * bbone_mult},
        {'params': hd, 'lr': peak_lr},
    ]


def build_lr_lambda(warmup_epochs, total_epochs, peak_lr, min_lr):
    """Linear warmup followed by cosine annealing."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr / peak_lr + 0.5 * (1.0 - min_lr / peak_lr) * (1 + np.cos(np.pi * t))
    return lr_lambda


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_epoch(model, loader, optimizer, scaler, hybridmix,
                device, accum_steps, criterion, autocast_ctx, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f'Ep {epoch:3d} Train', leave=False, ncols=110)
    for i, (imgs, lbls) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        imgs, ya, yb, lam = hybridmix(imgs, lbls)

        with autocast_ctx():
            out  = model(imgs)
            loss = (lam * criterion(out, ya) +
                    (1 - lam) * criterion(out, yb)) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        _, pred = out.max(1)
        total   += lbls.size(0)
        correct += pred.eq(ya).sum().item()
        pbar.set_postfix(loss=f'{total_loss/(i+1):.4f}',
                         acc=f'{100*correct/total:.2f}%')

    return total_loss / len(loader), correct / total


@torch.no_grad()
def validate(model, loader, device, criterion, autocast_ctx, epoch):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f'Ep {epoch:3d} Val  ', leave=False, ncols=110)
    for imgs, lbls in pbar:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        with autocast_ctx():
            out  = model(imgs)
            loss = criterion(out, lbls)
        total_loss += loss.item()
        _, pred = out.max(1)
        total   += lbls.size(0)
        correct += pred.eq(lbls).sum().item()
        pbar.set_postfix(acc=f'{100*correct/total:.2f}%')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return total_loss / len(loader), correct / total


@torch.no_grad()
def full_eval(model, loader, device, autocast_ctx):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for imgs, lbls in tqdm(loader, desc='  Evaluating', ncols=90):
        imgs = imgs.to(device, non_blocking=True)
        with autocast_ctx():
            logits = model(imgs)
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(lbls.numpy())
    return (np.vstack(all_probs),
            np.concatenate(all_preds),
            np.concatenate(all_labels))


@torch.no_grad()
def extract_features(model, loader, device, autocast_ctx, n_max=99999):
    model.eval()
    feats, lbls = [], []
    count = 0
    for imgs, lab in tqdm(loader, desc='  Feature extraction', ncols=90, leave=False):
        if count >= n_max:
            break
        imgs = imgs.to(device, non_blocking=True)
        with autocast_ctx():
            f = model.get_features(imgs)
        feats.append(f.cpu().numpy())
        lbls.append(lab.numpy())
        count += len(lab)
    return np.vstack(feats)[:n_max], np.concatenate(lbls)[:n_max]


@torch.no_grad()
def extract_tta_features(model, tta_dataset, device, autocast_ctx,
                          batch_size=4, num_workers=0):
    model.eval()
    ms     = MultiScaleExtractor(model)
    loader = DataLoader(tta_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)
    all_feats, all_labels = [], []
    for views, lbls in tqdm(loader, desc='  TTA features', ncols=90, leave=False):
        B, K, C, H, W = views.shape
        flat = views.view(B * K, C, H, W).to(device, non_blocking=True)
        ms._buf.clear()
        with autocast_ctx():
            f3 = model.backbone(flat)
        f2_raw   = ms._buf[0].to(f3.device)
        f2       = ms.pool(f2_raw).flatten(start_dim=1)
        combined = torch.cat([f2, f3], dim=1).cpu().numpy()
        combined = combined.reshape(B, K, -1).mean(axis=1)
        all_feats.append(combined)
        all_labels.append(lbls.numpy())
    ms.remove()
    return np.vstack(all_feats), np.concatenate(all_labels)


# =============================================================================
# SWIN PLOTS  (S1 - S6)
# =============================================================================

def plot_s1_training_history(history, best_val_acc, best_epoch, warmup_epochs, out_dir):
    epochs_ran = list(range(1, len(history['val_acc']) + 1))
    tr_acc_arr = np.array([v * 100 for v in history['train_acc']])
    vl_acc_arr = np.array([v * 100 for v in history['val_acc']])
    gap_arr    = tr_acc_arr - vl_acc_arr

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Swin Transformer - Training History\n'
        f'Best Val: {best_val_acc*100:.2f}%  |  Final Gap: {gap_arr[-1]:+.2f}%',
        fontsize=15, fontweight='bold',
    )

    ax = axes[0, 0]
    ax.plot(epochs_ran, tr_acc_arr, 'b-o', lw=2, ms=4, label='Train')
    ax.plot(epochs_ran, vl_acc_arr, color='orange', lw=2, ms=4,
            marker='s', label='Validation')
    ax.axhline(92, color='green', ls='--', lw=1.5, alpha=0.7, label='92% target')
    ax.axvline(warmup_epochs + 1, color='gray', ls=':', lw=1.5,
               alpha=0.6, label='Phase-2 start')
    ax.fill_between(epochs_ran, tr_acc_arr, vl_acc_arr,
                    where=(tr_acc_arr > vl_acc_arr),
                    alpha=0.15, color='red', label='Gap region')
    ax.set_title('Accuracy', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    if best_epoch <= len(epochs_ran):
        ax.annotate(
            f'Best: {best_val_acc*100:.2f}%',
            xy=(best_epoch, best_val_acc * 100),
            xytext=(min(best_epoch + 2, len(epochs_ran)), best_val_acc * 100 - 3),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=9, color='green',
        )

    ax = axes[0, 1]
    ax.plot(epochs_ran, history['train_loss'], 'b-o', lw=2, ms=4, label='Train')
    ax.plot(epochs_ran, history['val_loss'], color='orange', lw=2, ms=4,
            marker='s', label='Validation')
    ax.set_title('Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs_ran, gap_arr, lw=2, color='purple')
    ax.axhline(0,  color='black', lw=1, alpha=0.5)
    ax.axhline(2,  color='green', ls='--', lw=1.5, alpha=0.8, label='+-2% target')
    ax.axhline(-2, color='green', ls='--', lw=1.5, alpha=0.8)
    ax.fill_between(epochs_ran, 0, gap_arr,
                    where=(gap_arr > 0), color='red',  alpha=0.25, label='Train > Val')
    ax.fill_between(epochs_ran, 0, gap_arr,
                    where=(gap_arr < 0), color='blue', alpha=0.25, label='Val > Train')
    ax.set_title(f'Train-Val Gap  (Final: {gap_arr[-1]:+.2f}%)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap (%)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs_ran, history['lr'], lw=2, color='teal')
    ax.set_title('Learning Rate Schedule', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    savefig(out_dir / 'S1_train_val_history.png')


def plot_s2_confusion_matrix(val_labels, swin_preds, swin_acc, class_names, out_dir):
    cm      = confusion_matrix(val_labels, swin_preds)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Swin - Confusion Matrix  (Val Acc: {swin_acc*100:.2f}%)',
                 fontsize=14, fontweight='bold')

    sns.heatmap(cm_norm, ax=axes[0], cmap='Blues', vmin=0, vmax=1,
                xticklabels=False, yticklabels=False, cbar=True)
    axes[0].set_title('Normalised Confusion Matrix', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    per_cls_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-8) * 100
    top_i = np.argsort(per_cls_acc)[::-1][:25]
    axes[1].barh(range(len(top_i)), per_cls_acc[top_i], color='steelblue', alpha=0.75)
    axes[1].set_yticks(range(len(top_i)))
    axes[1].set_yticklabels([class_names[i][:22] for i in top_i], fontsize=8)
    axes[1].axvline(90, color='red', ls='--', lw=1.5, alpha=0.7, label='90%')
    axes[1].set_xlabel('Accuracy (%)')
    axes[1].set_title('Top-25 Classes by Accuracy', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='x')

    savefig(out_dir / 'S2_confusion_matrix.png')


def plot_s3_roc_curves(val_labels, swin_probs, class_names, num_classes, out_dir):
    freq        = np.bincount(val_labels, minlength=num_classes)
    top_classes = np.argsort(freq)[::-1][:12]
    val_bin     = label_binarize(val_labels, classes=list(range(num_classes)))
    cmap20      = plt.cm.get_cmap('tab20', 12)

    fig, ax = plt.subplots(figsize=(11, 8))
    for ki, cls_i in enumerate(top_classes):
        if cls_i >= swin_probs.shape[1]:
            continue
        fpr, tpr, _ = roc_curve(val_bin[:, cls_i], swin_probs[:, cls_i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.8, color=cmap20(ki),
                label=f'{class_names[cls_i][:28]} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title('Swin - ROC Curves (Top-12 Classes)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    savefig(out_dir / 'S3_roc_curves.png')


def plot_s4_f1_per_class(val_labels, swin_preds, class_names, logs_dir, out_dir):
    report = classification_report(
        val_labels, swin_preds,
        target_names=class_names, output_dict=True, zero_division=0,
    )
    rdf = pd.DataFrame(report).T
    rdf.to_csv(logs_dir / 'swin_classification_report.csv')

    cls_df = rdf.loc[class_names].copy()
    cls_df['f1-score'] = pd.to_numeric(cls_df['f1-score'], errors='coerce')
    cls_df = cls_df.sort_values('f1-score', ascending=False)
    top20  = cls_df.head(20)
    bot20  = cls_df.tail(20)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Swin - Per-Class F1 Score', fontsize=14, fontweight='bold')

    axes[0].barh(range(20), top20['f1-score'].values, color='seagreen', alpha=0.75)
    axes[0].set_yticks(range(20))
    axes[0].set_yticklabels([n[:25] for n in top20.index], fontsize=8)
    axes[0].set_xlabel('F1-Score')
    axes[0].set_title('Top-20 Classes', fontweight='bold')
    axes[0].axvline(0.9, color='red', ls='--', lw=1.5, alpha=0.7)
    axes[0].grid(alpha=0.3, axis='x')

    axes[1].barh(range(20), bot20['f1-score'].values, color='tomato', alpha=0.75)
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels([n[:25] for n in bot20.index], fontsize=8)
    axes[1].set_xlabel('F1-Score')
    axes[1].set_title('Bottom-20 Classes', fontweight='bold')
    axes[1].axvline(0.5, color='red', ls='--', lw=1.5, alpha=0.7)
    axes[1].grid(alpha=0.3, axis='x')

    savefig(out_dir / 'S4_f1_per_class.png')


def plot_s5_summary_metrics(swin_acc, prec_w, rec_w, f1_w,
                             prec_m, rec_m, f1_m, out_dir):
    m_names  = ['Accuracy', 'Prec\n(wt)', 'Recall\n(wt)',
                'F1\n(wt)', 'Prec\n(mac)', 'Recall\n(mac)', 'F1\n(mac)']
    m_values = [swin_acc, prec_w, rec_w, f1_w, prec_m, rec_m, f1_m]
    colors   = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2']

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(m_names, [v * 100 for v in m_values],
                  color=colors, alpha=0.80, edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, m_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f'{v*100:.2f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.axhline(92, color='red', ls='--', lw=1.5, alpha=0.7, label='92% target')
    ax.set_ylim([0, 107])
    ax.set_ylabel('Score (%)')
    ax.set_title('Swin - Summary Metrics', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.25, axis='y')
    savefig(out_dir / 'S5_summary_metrics.png')


def plot_s6_tsne(model, val_ds, device, autocast_ctx,
                  class_names, batch_size, num_workers, out_dir):
    tsne_loader = DataLoader(Subset(val_ds, list(range(2500))),
                             batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    ms_tsne = MultiScaleExtractor(model)
    tsne_raw, tsne_lbls = ms_tsne.extract(
        tsne_loader, device, autocast_ctx, desc='  S6 features')
    ms_tsne.remove()

    pca_vis = PCA(n_components=min(50, tsne_raw.shape[1]), random_state=42)
    tsne_xy = make_tsne(perplexity=40, n_iter=1000).fit_transform(
        pca_vis.fit_transform(tsne_raw))

    top15  = np.argsort(np.bincount(tsne_lbls))[::-1][:15]
    mask15 = np.isin(tsne_lbls, top15)
    cmap15 = plt.cm.get_cmap('tab20', 15)

    fig, ax = plt.subplots(figsize=(12, 9))
    for ki, cls_i in enumerate(top15):
        m = (tsne_lbls == cls_i) & mask15
        ax.scatter(tsne_xy[m, 0], tsne_xy[m, 1], s=18, alpha=0.75,
                   color=cmap15(ki), label=class_names[cls_i][:22])
    ax.set_title('Swin - t-SNE of Validation Features (Top-15)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.legend(loc='upper right', fontsize=8, ncol=2, markerscale=2)
    ax.grid(alpha=0.2)
    savefig(out_dir / 'S6_tsne.png')


# =============================================================================
# ML PLOTS  (M1 - M12)
# =============================================================================

def plot_m1_classifier_comparison(cv_results, val_accs, swin_acc, best_ind_acc, out_dir):
    all_names = list(cv_results.keys())
    cv_means  = [cv_results[n].mean() * 100 for n in all_names]
    cv_stds   = [cv_results[n].std()  * 100 for n in all_names]
    v_accs_p  = [val_accs.get(n, 0)   * 100 for n in all_names]
    x_pos     = np.arange(len(all_names))

    fig, ax = plt.subplots(figsize=(15, 6))
    b1 = ax.bar(x_pos - 0.22, cv_means, width=0.4, yerr=cv_stds, capsize=5,
                alpha=0.75, color='steelblue', label='5-Fold CV Acc')
    b2 = ax.bar(x_pos + 0.22, v_accs_p, width=0.4,
                alpha=0.75, color='coral', label='Val Acc (all)')
    for bar, v in zip(b2, v_accs_p):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.axhline(swin_acc * 100, color='green', ls='--', lw=2,
               label=f'Swin ({swin_acc*100:.2f}%)')
    ax.axhline(best_ind_acc * 100, color='red', ls='-.', lw=2,
               label=f'Best Individual ({best_ind_acc*100:.2f}%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_names, rotation=20, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('ML Classifiers - CV vs Val Accuracy  [multi-scale + TTA features]',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis='y')
    savefig(out_dir / 'M1_classifier_comparison.png')


def plot_m2_confusion_matrix(val_labels, best_ml_preds, best_ml_acc,
                               best_ml_name, class_names, out_dir):
    cm_ml      = confusion_matrix(val_labels, best_ml_preds)
    cm_ml_norm = cm_ml.astype(float) / (cm_ml.sum(axis=1, keepdims=True) + 1e-8)
    per_cls_acc = np.diag(cm_ml) / (cm_ml.sum(axis=1) + 1e-8) * 100

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'ML ({best_ml_name}) - Confusion  ({best_ml_acc*100:.2f}%)',
                 fontsize=14, fontweight='bold')

    sns.heatmap(cm_ml_norm, ax=axes[0], cmap='YlOrRd', vmin=0, vmax=1,
                xticklabels=False, yticklabels=False, cbar=True)
    axes[0].set_title('Normalised Confusion Matrix', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    top_i2 = np.argsort(per_cls_acc)[::-1][:25]
    axes[1].barh(range(len(top_i2)), per_cls_acc[top_i2], color='darkorange', alpha=0.75)
    axes[1].set_yticks(range(len(top_i2)))
    axes[1].set_yticklabels([class_names[i][:22] for i in top_i2], fontsize=8)
    axes[1].axvline(90, color='red', ls='--', lw=1.5, alpha=0.7, label='90%')
    axes[1].set_xlabel('Accuracy (%)')
    axes[1].set_title('Top-25 Classes by Accuracy', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='x')

    savefig(out_dir / 'M2_confusion_matrix.png')
    return per_cls_acc


def plot_m3_roc_curves(val_labels, best_ml_probs, best_ml_name,
                        class_names, num_classes, out_dir):
    if best_ml_probs is None:
        return
    freq        = np.bincount(val_labels, minlength=num_classes)
    top_classes = np.argsort(freq)[::-1][:12]
    val_bin     = label_binarize(val_labels, classes=list(range(num_classes)))
    cmap20      = plt.cm.get_cmap('tab20', 12)

    fig, ax = plt.subplots(figsize=(11, 8))
    for ki, cls_i in enumerate(top_classes):
        if cls_i >= best_ml_probs.shape[1]:
            continue
        fpr, tpr, _ = roc_curve(val_bin[:, cls_i], best_ml_probs[:, cls_i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.8, color=cmap20(ki),
                label=f'{class_names[cls_i][:28]} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title(f'ML ({best_ml_name}) - ROC Curves (Top-12)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    savefig(out_dir / 'M3_roc_curves.png')


def plot_m4_f1_per_class(val_labels, best_ml_preds, best_ml_acc,
                          best_ml_name, class_names, logs_dir, out_dir):
    ml_report = classification_report(
        val_labels, best_ml_preds,
        target_names=class_names, output_dict=True, zero_division=0)
    ml_rdf = pd.DataFrame(ml_report).T
    ml_rdf.to_csv(logs_dir / 'ml_classification_report.csv')
    ml_cls = ml_rdf.loc[class_names].copy()
    ml_cls['f1-score'] = pd.to_numeric(ml_cls['f1-score'], errors='coerce')
    ml_cls = ml_cls.sort_values('f1-score', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'ML ({best_ml_name}) - Per-Class F1  ({best_ml_acc*100:.2f}%)',
                 fontsize=14, fontweight='bold')

    for ax_, data, col, title, line in [
        (axes[0], ml_cls.head(20), 'mediumseagreen', 'Top-20',    0.9),
        (axes[1], ml_cls.tail(20), 'salmon',         'Bottom-20', 0.5),
    ]:
        ax_.barh(range(20), data['f1-score'].values, color=col, alpha=0.75)
        ax_.set_yticks(range(20))
        ax_.set_yticklabels([n[:25] for n in data.index], fontsize=8)
        ax_.set_xlabel('F1-Score')
        ax_.set_title(title, fontweight='bold')
        ax_.axvline(line, color='red', ls='--', lw=1.5, alpha=0.7)
        ax_.grid(alpha=0.3, axis='x')

    savefig(out_dir / 'M4_f1_per_class.png')


def plot_m5_summary_metrics(best_ml_acc, pml_w, rml_w, f1ml_w,
                             pml_m, rml_m, f1ml_m, best_ml_name, swin_acc, out_dir):
    m5n = ['Accuracy', 'Prec\n(wt)', 'Recall\n(wt)', 'F1\n(wt)',
           'Prec\n(mac)', 'Recall\n(mac)', 'F1\n(mac)']
    m5v = [best_ml_acc, pml_w, rml_w, f1ml_w, pml_m, rml_m, f1ml_m]
    c5  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
           '#9467bd', '#8c564b', '#e377c2']

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(m5n, [v * 100 for v in m5v], color=c5, alpha=0.80,
                  edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, m5v):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{v*100:.2f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.axhline(best_ml_acc * 100, color='red',   ls='--', lw=1.5, alpha=0.6,
               label=f'Best ML {best_ml_acc*100:.2f}%')
    ax.axhline(swin_acc * 100,    color='green', ls=':',  lw=1.5, alpha=0.6,
               label=f'Swin {swin_acc*100:.2f}%')
    ax.set_ylim([0, 107])
    ax.set_ylabel('Score (%)')
    ax.set_title(f'ML ({best_ml_name}) - Summary Metrics', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.25, axis='y')
    savefig(out_dir / 'M5_summary_metrics.png')


def plot_m6_tsne(val_feats_p, val_labels, class_names, out_dir):
    pca_vis    = PCA(n_components=min(50, val_feats_p.shape[1]), random_state=42)
    tsne_ml_xy = make_tsne(perplexity=40, n_iter=1000).fit_transform(
        pca_vis.fit_transform(val_feats_p[:2500]))
    tsne_ml_lb = val_labels[:2500]
    top15ml    = np.argsort(np.bincount(tsne_ml_lb))[::-1][:15]
    mask_ml    = np.isin(tsne_ml_lb, top15ml)
    cmap15     = plt.cm.get_cmap('tab20', 15)

    fig, ax = plt.subplots(figsize=(12, 9))
    for ki, cls_i in enumerate(top15ml):
        m = (tsne_ml_lb == cls_i) & mask_ml
        ax.scatter(tsne_ml_xy[m, 0], tsne_ml_xy[m, 1],
                   s=18, alpha=0.75, color=cmap15(ki),
                   label=class_names[cls_i][:22])
    ax.set_title('ML Features (PCA+TTA) - t-SNE (Top-15)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.legend(loc='upper right', fontsize=8, ncol=2, markerscale=2)
    ax.grid(alpha=0.2)
    savefig(out_dir / 'M6_tsne.png')


def plot_m7_swin_vs_ml(swin_acc, prec_w, rec_w, f1_w,
                        best_ml_acc, pml_w, rml_w, f1ml_w, best_ml_name, out_dir):
    sw_m   = [swin_acc, prec_w, rec_w, f1_w]
    ml_m_v = [best_ml_acc, pml_w, rml_w, f1ml_w]
    c_lbls = ['Accuracy', 'Precision\n(weighted)', 'Recall\n(weighted)', 'F1\n(weighted)']
    x      = np.arange(len(c_lbls))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w/2, [v * 100 for v in sw_m], w, alpha=0.8,
                color='steelblue', label=f'Swin ({swin_acc*100:.2f}%)')
    b2 = ax.bar(x + w/2, [v * 100 for v in ml_m_v], w, alpha=0.8,
                color='coral', label=f'{best_ml_name} ({best_ml_acc*100:.2f}%)')
    for bar, v in zip(list(b1) + list(b2), sw_m + ml_m_v):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{v*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(c_lbls)
    ax.set_ylim([0, 112])
    ax.set_ylabel('Score (%)')
    ax.grid(alpha=0.25, axis='y')
    ax.set_title('Swin vs Best ML - Performance Comparison',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    savefig(out_dir / 'M7_swin_vs_ml.png')


def plot_m8_blend_curve(swin_probs, ens_probs, swin_acc, ens_acc,
                         best_alpha, best_blend_acc, val_labels, out_dir):
    if ens_probs is None or np.array_equal(ens_probs, swin_probs):
        return
    alphas_v, blend_accs_v = [], []
    for alpha in np.arange(0.05, 0.97, 0.01):
        bp = alpha * swin_probs + (1.0 - alpha) * ens_probs
        alphas_v.append(alpha)
        blend_accs_v.append(accuracy_score(val_labels, bp.argmax(axis=1)) * 100)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(alphas_v, blend_accs_v, 'o-', lw=2, color='purple', ms=3)
    ax.axhline(swin_acc * 100, color='steelblue', ls='--', lw=1.5,
               label=f'Swin {swin_acc*100:.2f}%')
    ax.axhline(ens_acc  * 100, color='coral',     ls='--', lw=1.5,
               label=f'W-Ensemble {ens_acc*100:.2f}%')
    ax.axvline(best_alpha, color='green', ls=':', lw=2,
               label=f'Best alpha={best_alpha:.2f}  ->  {best_blend_acc*100:.2f}%')
    ax.set_xlabel('Swin weight (alpha)')
    ax.set_ylabel('Val Accuracy (%)')
    ax.set_title('Swin + Weighted Ensemble Blend - Accuracy vs Mix Ratio',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    savefig(out_dir / 'M8_blend_curve.png')


def plot_m9_all_candidates(candidates, swin_acc, out_dir):
    cand_names = list(candidates.keys())
    cand_accs  = [candidates[n][0] * 100 for n in cand_names]
    sorted_idx = np.argsort(cand_accs)[::-1]
    names_s    = [cand_names[i] for i in sorted_idx]
    accs_s     = [cand_accs[i]  for i in sorted_idx]

    colors = []
    for n in names_s:
        if n.startswith('Blend'):    colors.append('#9467bd')
        elif n == 'Stacking':        colors.append('#d62728')
        elif 'Ensemble' in n:        colors.append('#ff7f0e')
        else:                        colors.append('#1f77b4')

    fig, ax = plt.subplots(figsize=(14, max(5, len(names_s) * 0.5)))
    bars = ax.barh(range(len(names_s)), accs_s, color=colors, alpha=0.80)
    for bar, v in zip(bars, accs_s):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{v:.2f}%', va='center', ha='left', fontsize=9, fontweight='bold')
    ax.set_yticks(range(len(names_s)))
    ax.set_yticklabels(names_s, fontsize=9)
    ax.axvline(swin_acc * 100, color='green', ls='--', lw=2,
               label=f'Swin {swin_acc*100:.2f}%')
    ax.set_xlabel('Val Accuracy (%)')
    ax.set_title('All Candidates - Accuracy Race',
                 fontsize=12, fontweight='bold')
    legend_els = [
        Patch(color='#1f77b4', label='Individual'),
        Patch(color='#ff7f0e', label='Ensemble'),
        Patch(color='#d62728', label='Stacking'),
        Patch(color='#9467bd', label='Blend'),
        Patch(color='green',   label=f'Swin {swin_acc*100:.2f}%'),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc='lower right')
    ax.grid(alpha=0.3, axis='x')
    savefig(out_dir / 'M9_all_candidates.png')


def plot_m10_calibration(val_labels, best_ml_preds, best_ml_probs,
                          best_ml_name, freq, class_names, out_dir):
    if best_ml_probs is None:
        return
    top5   = np.argsort(freq)[::-1][:5]
    cmap20 = plt.cm.get_cmap('tab20', 12)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'ML ({best_ml_name}) - Probability Calibration',
                 fontsize=13, fontweight='bold')

    ax_cal = axes[0]
    for ki, cls_i in enumerate(top5):
        y_true_bin = (val_labels == cls_i).astype(int)
        y_prob_cls = best_ml_probs[:, cls_i]
        frac_pos, mean_pred = calibration_curve(
            y_true_bin, y_prob_cls, n_bins=15, strategy='uniform')
        ax_cal.plot(mean_pred, frac_pos, 's-', lw=1.8,
                    color=cmap20(ki), label=class_names[cls_i][:20])
    ax_cal.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect')
    ax_cal.set_xlabel('Mean Predicted Probability')
    ax_cal.set_ylabel('Fraction of Positives')
    ax_cal.set_title('Reliability Diagram (Top-5 Classes)')
    ax_cal.legend(fontsize=8)
    ax_cal.grid(alpha=0.3)

    ax_conf   = axes[1]
    max_probs = best_ml_probs.max(axis=1)
    correct   = (best_ml_preds == val_labels)
    ax_conf.hist(max_probs[correct],  bins=40, alpha=0.6, color='green', label='Correct')
    ax_conf.hist(max_probs[~correct], bins=40, alpha=0.6, color='red',   label='Incorrect')
    ax_conf.set_xlabel('Max Predicted Probability')
    ax_conf.set_ylabel('Count')
    ax_conf.set_title('Confidence Distribution')
    ax_conf.legend(fontsize=10)
    ax_conf.grid(alpha=0.3)

    savefig(out_dir / 'M10_calibration.png')


def plot_m11_per_class_delta(val_labels, swin_preds, best_ml_preds,
                              best_ml_name, class_names, out_dir):
    cm_swin      = confusion_matrix(val_labels, swin_preds)
    swin_per_cls = np.diag(cm_swin) / (cm_swin.sum(axis=1) + 1e-8) * 100
    cm_ml        = confusion_matrix(val_labels, best_ml_preds)
    ml_per_cls   = np.diag(cm_ml)  / (cm_ml.sum(axis=1)   + 1e-8) * 100
    delta_cls    = ml_per_cls - swin_per_cls
    sort_delta   = np.argsort(delta_cls)[::-1]
    top30_gain   = sort_delta[:30]
    bot30_gain   = sort_delta[-30:]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'Per-Class Accuracy: {best_ml_name} vs Swin  (Delta = ML - Swin)',
                 fontsize=13, fontweight='bold')

    for ax_, idxs, title, col_pos, col_neg in [
        (axes[0], top30_gain, 'Top-30 Gain Classes',  'mediumseagreen', 'salmon'),
        (axes[1], bot30_gain, 'Bottom-30 (ML Loses)', 'salmon',         'mediumseagreen'),
    ]:
        deltas = delta_cls[idxs]
        colors = [col_pos if d >= 0 else col_neg for d in deltas]
        ax_.barh(range(len(idxs)), deltas, color=colors, alpha=0.80)
        ax_.set_yticks(range(len(idxs)))
        ax_.set_yticklabels([class_names[i][:25] for i in idxs], fontsize=7)
        ax_.axvline(0, color='black', lw=1.5)
        ax_.set_xlabel('Delta Accuracy (ML - Swin) %')
        ax_.set_title(title, fontweight='bold')
        ax_.grid(alpha=0.3, axis='x')

    savefig(out_dir / 'M11_per_class_delta.png')


def plot_m12_stacking_weights(meta_learner, stacking_members, trained,
                               val_feats_p, val_labels, best_ml_preds,
                               best_ind_name, val_accs, class_names, out_dir):
    if meta_learner is None or not hasattr(meta_learner, 'coef_'):
        return
    coef      = np.abs(meta_learner.coef_)
    n_members = len(stacking_members)
    chunk     = coef.shape[1] // n_members

    member_weights = []
    for mi in range(n_members):
        start = mi * chunk
        end   = start + chunk
        member_weights.append(coef[:, start:end].mean())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Stacking Meta-Learner - Classifier Contribution',
                 fontsize=13, fontweight='bold')

    colors_st = plt.cm.get_cmap('Set2', len(stacking_members))
    axes[0].barh(stacking_members, member_weights,
                 color=[colors_st(i) for i in range(len(stacking_members))],
                 alpha=0.85)
    axes[0].set_xlabel('Mean |coef| in Meta-Learner')
    axes[0].set_title('Per-Classifier Weight')
    axes[0].grid(alpha=0.3, axis='x')

    if best_ind_name in val_accs:
        ind_preds = trained[best_ind_name].predict(val_feats_p)
        cm_ind    = confusion_matrix(val_labels, ind_preds)
        ind_per   = np.diag(cm_ind) / (cm_ind.sum(axis=1) + 1e-8) * 100
        cm_ml     = confusion_matrix(val_labels, best_ml_preds)
        per_cls_acc_best = np.diag(cm_ml) / (cm_ml.sum(axis=1) + 1e-8) * 100
        st_delta  = per_cls_acc_best - ind_per
        top20_st  = np.argsort(np.abs(st_delta))[::-1][:20]
        col_st    = ['mediumseagreen' if st_delta[i] >= 0 else 'salmon' for i in top20_st]
        axes[1].barh(range(20), st_delta[top20_st], color=col_st, alpha=0.80)
        axes[1].set_yticks(range(20))
        axes[1].set_yticklabels([class_names[i][:25] for i in top20_st], fontsize=8)
        axes[1].axvline(0, color='black', lw=1.5)
        axes[1].set_xlabel(f'Delta Accuracy (Best ML - {best_ind_name}) %')
        axes[1].set_title(f'Top-20 Class Delta: Best vs {best_ind_name}')
        axes[1].grid(alpha=0.3, axis='x')

    savefig(out_dir / 'M12_stacking_weights.png')


# =============================================================================
# BLEND SWEEP
# =============================================================================

def blend_sweep(swin_p, ml_p, ml_name, val_labels, swin_acc,
                alpha_min=0.10, alpha_max=0.95):
    best_acc, best_a = 0.0, 0.5
    for a in np.arange(alpha_min, alpha_max, 0.01):
        bp   = a * swin_p + (1.0 - a) * ml_p
        bacc = accuracy_score(val_labels, bp.argmax(axis=1))
        if bacc > best_acc:
            best_acc = bacc
            best_a   = float(a)
    blend_p    = best_a * swin_p + (1.0 - best_a) * ml_p
    blend_pred = blend_p.argmax(axis=1)
    beat       = "BEATS SWIN" if best_acc > swin_acc else ""
    print(f"  Swin+{ml_name:<14}: {best_acc*100:.2f}%  "
          f"(alpha={best_a:.2f}, Delta {(best_acc-swin_acc)*100:+.2f}%)  {beat}")
    return best_acc, best_a, blend_p, blend_pred


# =============================================================================
# MAIN  -  Windows multiprocessing requires this guard
# =============================================================================
if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------------

    DATASET_PATH = r"Provide the dataset path"

    IMG_SIZE         = 224
    BATCH_SIZE       = 16
    ACCUM_STEPS      = 4          # Effective batch size = 64
    NUM_WORKERS      = 0          # Must be 0 on Windows
    TOTAL_EPOCHS     = 30
    WARMUP_EPOCHS    = 4
    PEAK_LR          = 8e-5
    BACKBONE_LR_MULT = 0.08
    MIN_LR           = 1e-6
    WEIGHT_DECAY     = 0.02
    LABEL_SMOOTHING  = 0.05
    DROPOUT          = 0.25
    MIXUP_PROB       = 0.12
    PATIENCE         = 7
    ML_SUBSET        = 6000
    PCA_DIMS         = 768

    USE_TTA          = True
    REUSE_SWIN       = True   # Load cached Swin predictions if they exist
    REUSE_FEATURES   = True   # Load cached ML features if they exist
    SKIP_XGB         = False
    SKIP_KNN         = False

    STACK_SUBSAMPLE  = 14000
    STACK_CV_FOLDS   = 5
    SKIP_FROM_STACK  = {'XGBoost', 'KNN', 'SVM-RBF-C50'}

    DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autocast_ctx = _make_autocast()
    scaler       = _make_scaler()

    BASE_OUT   = Path('.')
    PLOTS_SWIN = BASE_OUT / 'plots_swin'
    PLOTS_ML   = BASE_OUT / 'plots_ml'
    MODELS_DIR = BASE_OUT / 'models'
    LOGS_DIR   = BASE_OUT / 'logs'
    for d in [PLOTS_SWIN, PLOTS_ML, MODELS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    print("=" * 72)
    print("  TCMP-300  Combined Pipeline")
    print("  Target: Val > 92%  |  Train-Val Gap < 2%  |  ML > 94-95%")
    print("=" * 72)
    print(f"  Device      : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"  Batch size  : {BATCH_SIZE} x {ACCUM_STEPS} = {BATCH_SIZE * ACCUM_STEPS} (effective)")
    print(f"  Epochs      : {TOTAL_EPOCHS}  |  Warmup: {WARMUP_EPOCHS}")
    print(f"  NUM_WORKERS : {NUM_WORKERS}")
    print(f"  TTA         : {USE_TTA}")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # PHASE 1 : DATA LOADING
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  LOADING DATASET")
    print("-" * 60)

    val_tfm = BalancedAug.val(IMG_SIZE)

    train_ds = datasets.ImageFolder(
        root=Path(DATASET_PATH) / 'train',
        transform=BalancedAug.train(IMG_SIZE),
        loader=safe_loader,
    )
    train_ds_feat = datasets.ImageFolder(
        root=Path(DATASET_PATH) / 'train',
        transform=val_tfm,
        loader=safe_loader,
    )
    val_ds = datasets.ImageFolder(
        root=Path(DATASET_PATH) / 'val',
        transform=val_tfm,
        loader=safe_loader,
    )

    NUM_CLASSES = len(train_ds.classes)
    class_names = train_ds.classes

    with open(MODELS_DIR / 'class_names.json', 'w') as f:
        json.dump(class_names, f)

    targets = np.array(train_ds.targets)
    indices = []
    per_cls = max(1, ML_SUBSET // NUM_CLASSES)
    for c in range(NUM_CLASSES):
        cidx = np.where(targets == c)[0]
        n    = min(per_cls, len(cidx))
        indices.extend(random.sample(list(cidx), n))
    random.shuffle(indices)
    train_ml_ds = Subset(train_ds, indices)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    ml_loader = DataLoader(
        train_ml_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    train_loader_feat = DataLoader(
        train_ds_feat, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f"  Classes    : {NUM_CLASSES}")
    print(f"  Train      : {len(train_ds):,}")
    print(f"  Val        : {len(val_ds):,}")
    print(f"  ML subset  : {len(train_ml_ds):,}")

    # -------------------------------------------------------------------------
    # PHASE 1 : MODEL + OPTIMIZER
    # -------------------------------------------------------------------------
    model     = SwinClassifier(num_classes=NUM_CLASSES, dropout=DROPOUT,
                               pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    hybridmix = HybridMix(prob=MIXUP_PROB)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {total_p:,}")

    optimizer = optim.AdamW(
        make_param_groups(model, PEAK_LR, BACKBONE_LR_MULT, WEIGHT_DECAY),
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8,
    )
    lr_lambda = build_lr_lambda(WARMUP_EPOCHS, TOTAL_EPOCHS, PEAK_LR, MIN_LR)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for n, p in model.backbone.named_parameters():
        if 'patch_embed' in n:
            p.requires_grad = False
        elif n.startswith('layers.'):
            try:
                stage = int(n.split('.')[1])
                if stage < 2:
                    p.requires_grad = False
            except (IndexError, ValueError):
                p.requires_grad = False

    # -------------------------------------------------------------------------
    # PHASE 1 : TRAINING LOOP
    # -------------------------------------------------------------------------
    history      = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[], lr=[])
    best_val_acc = 0.0
    best_epoch   = 0
    no_improve   = 0

    print("\n" + "=" * 72)
    print("  TRAINING  (Target: Val > 92%  |  Gap < 2%)")
    print("=" * 72)
    print(f"{'Ep':>4} {'Train%':>8} {'Val%':>8} {'Gap%':>7} {'LR':>10}  Status")
    print("-" * 72)

    for epoch in range(1, TOTAL_EPOCHS + 1):

        if epoch == WARMUP_EPOCHS + 1:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(
                make_param_groups(model, PEAK_LR, BACKBONE_LR_MULT, WEIGHT_DECAY),
                weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8,
            )
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            for _ in range(epoch - 1):
                scheduler.step()
            print(f"\n  Phase-2: Full fine-tune (epoch {epoch})\n")

        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scaler, hybridmix,
            DEVICE, ACCUM_STEPS, criterion, autocast_ctx, epoch,
        )
        vl_loss, vl_acc = validate(
            model, val_loader, DEVICE, criterion, autocast_ctx, epoch,
        )
        scheduler.step()

        lr_now = optimizer.param_groups[-1]['lr']
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)
        history['lr'].append(lr_now)

        gap     = (tr_acc - vl_acc) * 100
        is_best = vl_acc > best_val_acc
        if is_best:
            best_val_acc, best_epoch = vl_acc, epoch
            no_improve = 0
            torch.save({
                'epoch': epoch, 'val_acc': vl_acc,
                'model_state_dict': model.state_dict(),
                'num_classes': NUM_CLASSES,
            }, MODELS_DIR / 'swin_best.pth')
        else:
            no_improve += 1

        if abs(gap) < 2:
            status = 'OK <2%'
        elif abs(gap) < 5:
            status = 'WARN  '
        else:
            status = 'LARGE '
        star = ' BEST' if is_best else '     '
        print(f"{epoch:4d} {tr_acc*100:8.2f} {vl_acc*100:8.2f}{star} "
              f"{gap:+7.2f} {lr_now:10.2e}  {status}  ({time.time()-t0:.0f}s)")

        if no_improve >= PATIENCE and epoch > WARMUP_EPOCHS + 3:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print("-" * 72)
    print(f"\n  Best Val: {best_val_acc*100:.2f}%  @  epoch {best_epoch}")

    with open(LOGS_DIR / 'swin_history.json', 'w') as f:
        json.dump(history, f)

    # -------------------------------------------------------------------------
    # PHASE 1 : LOAD BEST MODEL
    # -------------------------------------------------------------------------
    ckpt = torch.load(MODELS_DIR / 'swin_best.pth',
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"\n  Loaded best checkpoint - epoch {ckpt['epoch']}  "
          f"Val {ckpt['val_acc']*100:.2f}%")

    # -------------------------------------------------------------------------
    # PHASE 1 : SWIN FULL EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  SWIN FULL EVALUATION")
    print("-" * 60)

    _swin_probs_f  = LOGS_DIR / 'swin_val_probs.npy'
    _swin_preds_f  = LOGS_DIR / 'swin_val_preds.npy'
    _swin_labels_f = LOGS_DIR / 'swin_val_labels.npy'

    if REUSE_SWIN and _swin_probs_f.exists():
        print("  Loading cached Swin predictions...")
        swin_probs = np.load(_swin_probs_f)
        swin_preds = np.load(_swin_preds_f)
        val_labels = np.load(_swin_labels_f)
    else:
        swin_probs, swin_preds, val_labels = full_eval(
            model, val_loader, DEVICE, autocast_ctx)
        np.save(_swin_probs_f, swin_probs)
        np.save(_swin_preds_f, swin_preds)
        np.save(_swin_labels_f, val_labels)

    swin_acc = accuracy_score(val_labels, swin_preds)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        val_labels, swin_preds, average='weighted', zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        val_labels, swin_preds, average='macro',    zero_division=0)

    print(f"\n  Accuracy            : {swin_acc*100:.2f}%")
    print(f"  Precision (weighted): {prec_w:.4f}")
    print(f"  Recall    (weighted): {rec_w:.4f}")
    print(f"  F1-Score  (weighted): {f1_w:.4f}")
    print(f"  Precision (macro)   : {prec_m:.4f}")
    print(f"  Recall    (macro)   : {rec_m:.4f}")
    print(f"  F1-Score  (macro)   : {f1_m:.4f}")

    # -------------------------------------------------------------------------
    # PHASE 1 : SWIN PLOTS  (S1 - S5)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  SWIN PLOTS  (S1 - S5)")
    print("-" * 60)

    plot_s1_training_history(history, best_val_acc, best_epoch, WARMUP_EPOCHS, PLOTS_SWIN)
    plot_s2_confusion_matrix(val_labels, swin_preds, swin_acc, class_names, PLOTS_SWIN)
    plot_s3_roc_curves(val_labels, swin_probs, class_names, NUM_CLASSES, PLOTS_SWIN)
    plot_s4_f1_per_class(val_labels, swin_preds, class_names, LOGS_DIR, PLOTS_SWIN)
    plot_s5_summary_metrics(swin_acc, prec_w, rec_w, f1_w, prec_m, rec_m, f1_m, PLOTS_SWIN)

    # -------------------------------------------------------------------------
    # PHASE 2 : S6 t-SNE  (skip if already saved)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  SWIN t-SNE  (S6)")
    print("-" * 60)

    if not (PLOTS_SWIN / 'S6_tsne.png').exists():
        plot_s6_tsne(model, val_ds, DEVICE, autocast_ctx,
                     class_names, BATCH_SIZE, NUM_WORKERS, PLOTS_SWIN)
    else:
        print("  S6 t-SNE already exists, skipping.")

    # -------------------------------------------------------------------------
    # PHASE 2 : FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  FEATURE EXTRACTION  (multi-scale + TTA)")
    print("-" * 60)

    _tr_feats_f = LOGS_DIR / 'ml_train_feats.npy'
    _tr_lbls_f  = LOGS_DIR / 'ml_train_labels.npy'
    _vl_feats_f = LOGS_DIR / 'ml_val_feats.npy'

    if REUSE_FEATURES and _tr_feats_f.exists() and _vl_feats_f.exists():
        print("  Loading cached PCA features...")
        ml_feats_p    = np.load(_tr_feats_f)
        ml_feats_lbls = np.load(_tr_lbls_f)
        val_feats_p   = np.load(_vl_feats_f)
        print(f"  Train: {ml_feats_p.shape}")
        print(f"  Val  : {val_feats_p.shape}")
    else:
        ms_tr = MultiScaleExtractor(model)
        ml_feats_raw, ml_feats_lbls = ms_tr.extract(
            train_loader_feat, DEVICE, autocast_ctx, desc='  Train features')
        ms_tr.remove()

        if USE_TTA:
            print("  Val TTA extraction (10 views/image)...")
            tta_ds = TTADataset(val_ds, img_size=IMG_SIZE)
            val_feats_raw, _vl = extract_tta_features(
                model, tta_ds, DEVICE, autocast_ctx,
                batch_size=4, num_workers=NUM_WORKERS)
        else:
            ms_v = MultiScaleExtractor(model)
            val_feats_raw, _vl = ms_v.extract(
                val_loader, DEVICE, autocast_ctx, desc='  Val features')
            ms_v.remove()

        assert np.array_equal(_vl, val_labels), \
            f"Val label mismatch: extracted {len(_vl)}, expected {len(val_labels)}"

        print(f"\n  Train raw : {ml_feats_raw.shape}")
        print(f"  Val raw   : {val_feats_raw.shape}")

        feat_scaler = StandardScaler()
        pca         = PCA(n_components=min(PCA_DIMS, ml_feats_raw.shape[1]), random_state=42)
        ml_feats_p  = pca.fit_transform(feat_scaler.fit_transform(ml_feats_raw))
        val_feats_p = pca.transform(feat_scaler.transform(val_feats_raw))

        print(f"  After PCA : {ml_feats_p.shape}  /  {val_feats_p.shape}")
        print(f"  PCA var   : {pca.explained_variance_ratio_.sum()*100:.1f}%")

        np.save(_tr_feats_f,  ml_feats_p)
        np.save(_tr_lbls_f,   ml_feats_lbls)
        np.save(_vl_feats_f,  val_feats_p)

    # -------------------------------------------------------------------------
    # PHASE 2 : ML CLASSIFIERS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"  ML CLASSIFIERS  (target > {swin_acc*100:.2f}%  ->  94-95%)")
    print("-" * 60)

    _xgb_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classifiers = {
        'SVM-RBF': SVC(
            kernel='rbf', C=25.0, gamma='scale',
            probability=True, random_state=42, class_weight='balanced',
        ),
        'SVM-RBF-C50': SVC(
            kernel='rbf', C=50.0, gamma='scale',
            probability=True, random_state=42, class_weight='balanced',
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=600, max_features='sqrt',
            min_samples_leaf=1, random_state=42,
            n_jobs=-1, class_weight='balanced',
        ),
        'LogisticReg': LogisticRegression(
            C=10.0, max_iter=2000, solver='lbfgs',
            n_jobs=-1, random_state=42,
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(2048, 1024, 512),
            activation='relu', solver='adam',
            alpha=3e-4, learning_rate='adaptive',
            learning_rate_init=1e-3,
            max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.08,
            n_iter_no_change=25, batch_size=512,
        ),
    }

    if not SKIP_XGB:
        classifiers['XGBoost'] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            min_child_weight=2, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, eval_metric='mlogloss',
            tree_method='hist', device=_xgb_device,
        )

    if not SKIP_KNN:
        classifiers['KNN'] = KNeighborsClassifier(
            n_neighbors=7, metric='cosine', n_jobs=-1,
        )

    if HAS_LGBM:
        classifiers['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=800, num_leaves=255, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            min_child_samples=3, reg_alpha=0.05, reg_lambda=0.5,
            random_state=42, class_weight='balanced',
            n_jobs=-1, verbose=-1,
        )

    cv_n   = min(len(ml_feats_p), STACK_SUBSAMPLE)
    rng42  = np.random.RandomState(42)
    cv_idx = rng42.choice(len(ml_feats_p), cv_n, replace=False)
    cv_X   = ml_feats_p[cv_idx]
    cv_y   = ml_feats_lbls[cv_idx]

    skf = StratifiedKFold(n_splits=STACK_CV_FOLDS, shuffle=True, random_state=42)

    cv_results = {}
    trained    = {}
    val_accs   = {}
    oof_probs  = {}

    for name, clf in classifiers.items():
        print(f"\n  [{name}]")
        t0 = time.time()
        try:
            fold_accs = []
            do_stack  = (name not in SKIP_FROM_STACK) and hasattr(clf, 'predict_proba')
            oof_p     = np.zeros((cv_n, NUM_CLASSES)) if do_stack else None

            for fold_i, (tr_i, val_i) in enumerate(skf.split(cv_X, cv_y)):
                clf_fold = clone(clf)
                clf_fold.fit(cv_X[tr_i], cv_y[tr_i])
                fold_pred = clf_fold.predict(cv_X[val_i])
                fold_accs.append(accuracy_score(cv_y[val_i], fold_pred))
                if do_stack:
                    oof_p[val_i] = clf_fold.predict_proba(cv_X[val_i])

            cv_results[name] = np.array(fold_accs)
            if do_stack:
                oof_probs[name] = oof_p
            print(f"    5-Fold CV  : {np.mean(fold_accs)*100:.2f} "
                  f"+/- {np.std(fold_accs)*100:.2f}%")

            clf.fit(ml_feats_p, ml_feats_lbls)
            trained[name] = clf

            preds = clf.predict(val_feats_p)
            acc   = accuracy_score(val_labels, preds)
            val_accs[name] = acc
            delta = (acc - swin_acc) * 100
            beat  = "BEATS SWIN" if acc > swin_acc else ""
            print(f"    Val Acc    : {acc*100:.2f}%  "
                  f"(Delta {delta:+.2f}%)  {beat}  [{time.time()-t0:.0f}s]")

        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            cv_results[name] = np.array([0.0])

    # -------------------------------------------------------------------------
    # PHASE 2 : STACKING META-LEARNER
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  STACKING META-LEARNER")
    print("-" * 60)

    stack_acc    = 0.0
    stack_preds  = None
    stack_probs  = None
    meta_learner = None

    stacking_members = [n for n in oof_probs if n in trained]
    print(f"  OOF members: {stacking_members}")

    if len(stacking_members) >= 2:
        meta_X_train = np.hstack([oof_probs[n] for n in stacking_members])
        meta_y_train = cv_y

        val_meta_parts = [trained[n].predict_proba(val_feats_p) for n in stacking_members]
        meta_X_val     = np.hstack(val_meta_parts)

        meta_learner = LogisticRegression(
            C=0.05, max_iter=500, solver='lbfgs',
            n_jobs=-1, random_state=42,
        )
        t0 = time.time()
        meta_learner.fit(meta_X_train, meta_y_train)
        stack_preds = meta_learner.predict(meta_X_val)
        stack_probs = meta_learner.predict_proba(meta_X_val)
        stack_acc   = accuracy_score(val_labels, stack_preds)
        delta_st    = (stack_acc - swin_acc) * 100
        beat_st     = "BEATS SWIN" if stack_acc > swin_acc else ""
        print(f"  Stacking Val Acc : {stack_acc*100:.2f}%  "
              f"(Delta {delta_st:+.2f}%)  {beat_st}  [{time.time()-t0:.0f}s]")
    else:
        print("  Not enough OOF classifiers for stacking (need >= 2).")

    # -------------------------------------------------------------------------
    # PHASE 2 : WEIGHTED ENSEMBLE + BLEND
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  WEIGHTED ENSEMBLE + BLEND")
    print("-" * 60)

    proba_trained = {n: clf for n, clf in trained.items()
                     if hasattr(clf, 'predict_proba')}

    if proba_trained:
        w_sum = sum(val_accs[n] ** 2 for n in proba_trained)
        w_ens_probs = sum(
            (val_accs[n] ** 2 / w_sum) * trained[n].predict_proba(val_feats_p)
            for n in proba_trained
        )
        w_ens_preds = w_ens_probs.argmax(axis=1)
        w_ens_acc   = accuracy_score(val_labels, w_ens_preds)
        print(f"  Weighted Ensemble: {w_ens_acc*100:.2f}%  "
              f"(Delta {(w_ens_acc-swin_acc)*100:+.2f}%)"
              f"  {'BEATS SWIN' if w_ens_acc > swin_acc else ''}")
    else:
        w_ens_probs = None
        w_ens_preds = None
        w_ens_acc   = 0.0

    if proba_trained:
        u_ens_probs = np.mean(
            [trained[n].predict_proba(val_feats_p) for n in proba_trained], axis=0)
        u_ens_preds = u_ens_probs.argmax(axis=1)
        u_ens_acc   = accuracy_score(val_labels, u_ens_preds)
        print(f"  Uniform Ensemble : {u_ens_acc*100:.2f}%  "
              f"(Delta {(u_ens_acc-swin_acc)*100:+.2f}%)")
    else:
        u_ens_probs = w_ens_probs
        u_ens_preds = w_ens_preds
        u_ens_acc   = w_ens_acc

    best_ind_name = max(val_accs, key=val_accs.get) if val_accs else None
    best_ind_acc  = val_accs[best_ind_name] if best_ind_name else 0.0
    print(f"\n  Best individual  : {best_ind_name}  ->  {best_ind_acc*100:.2f}%")

    blend_results = {}

    if w_ens_probs is not None:
        r = blend_sweep(swin_probs, w_ens_probs, 'WeightedEns', val_labels, swin_acc)
        blend_results['Blend_wEns'] = r

    if stack_probs is not None:
        r = blend_sweep(swin_probs, stack_probs, 'Stacking', val_labels, swin_acc)
        blend_results['Blend_Stack'] = r

    if best_ind_name and hasattr(trained.get(best_ind_name), 'predict_proba'):
        ind_probs = trained[best_ind_name].predict_proba(val_feats_p)
        r = blend_sweep(swin_probs, ind_probs, best_ind_name, val_labels, swin_acc)
        blend_results[f'Blend_{best_ind_name}'] = r

    # -------------------------------------------------------------------------
    # PHASE 2 : PICK OVERALL BEST
    # -------------------------------------------------------------------------
    candidates = {}
    for n, clf in trained.items():
        preds = clf.predict(val_feats_p)
        acc   = val_accs[n]
        prbs  = clf.predict_proba(val_feats_p) if hasattr(clf, 'predict_proba') else None
        candidates[n] = (acc, preds, prbs)

    if w_ens_probs is not None:
        candidates['WeightedEnsemble'] = (w_ens_acc, w_ens_preds, w_ens_probs)
    if u_ens_probs is not None:
        candidates['UniformEnsemble']  = (u_ens_acc, u_ens_preds, u_ens_probs)
    if stack_probs is not None:
        candidates['Stacking']         = (stack_acc, stack_preds, stack_probs)
    for bname, (bacc, ba, bprobs, bpreds) in blend_results.items():
        candidates[bname] = (bacc, bpreds, bprobs)

    best_ml_name  = max(candidates, key=lambda k: candidates[k][0])
    best_ml_acc, best_ml_preds, best_ml_probs = candidates[best_ml_name]

    pml_w, rml_w, f1ml_w, _ = precision_recall_fscore_support(
        val_labels, best_ml_preds, average='weighted', zero_division=0)
    pml_m, rml_m, f1ml_m, _ = precision_recall_fscore_support(
        val_labels, best_ml_preds, average='macro',    zero_division=0)

    print(f"\n  BEST OVERALL : {best_ml_name}  ->  {best_ml_acc*100:.2f}%")

    ens_probs        = w_ens_probs if w_ens_probs is not None else swin_probs
    ens_acc          = w_ens_acc
    best_alpha       = blend_results.get('Blend_wEns', (0, 0.5))[1] \
                       if 'Blend_wEns' in blend_results else 0.5
    best_blend_acc   = blend_results.get('Blend_wEns', (swin_acc,))[0] \
                       if 'Blend_wEns' in blend_results else swin_acc
    freq             = np.bincount(val_labels, minlength=NUM_CLASSES)

    # -------------------------------------------------------------------------
    # PHASE 2 : ML PLOTS  (M1 - M12)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  ML PLOTS  (M1 - M12)")
    print("-" * 60)

    plot_m1_classifier_comparison(cv_results, val_accs, swin_acc, best_ind_acc, PLOTS_ML)
    plot_m2_confusion_matrix(val_labels, best_ml_preds, best_ml_acc,
                              best_ml_name, class_names, PLOTS_ML)
    plot_m3_roc_curves(val_labels, best_ml_probs, best_ml_name,
                        class_names, NUM_CLASSES, PLOTS_ML)
    plot_m4_f1_per_class(val_labels, best_ml_preds, best_ml_acc,
                          best_ml_name, class_names, LOGS_DIR, PLOTS_ML)
    plot_m5_summary_metrics(best_ml_acc, pml_w, rml_w, f1ml_w,
                             pml_m, rml_m, f1ml_m, best_ml_name, swin_acc, PLOTS_ML)

    print("  Computing M6 t-SNE...")
    plot_m6_tsne(val_feats_p, val_labels, class_names, PLOTS_ML)

    plot_m7_swin_vs_ml(swin_acc, prec_w, rec_w, f1_w,
                        best_ml_acc, pml_w, rml_w, f1ml_w, best_ml_name, PLOTS_ML)
    plot_m8_blend_curve(swin_probs, ens_probs, swin_acc, ens_acc,
                         best_alpha, best_blend_acc, val_labels, PLOTS_ML)

    print("  Plotting M9...")
    plot_m9_all_candidates(candidates, swin_acc, PLOTS_ML)

    print("  Plotting M10...")
    plot_m10_calibration(val_labels, best_ml_preds, best_ml_probs,
                          best_ml_name, freq, class_names, PLOTS_ML)

    print("  Plotting M11...")
    plot_m11_per_class_delta(val_labels, swin_preds, best_ml_preds,
                              best_ml_name, class_names, PLOTS_ML)

    print("  Plotting M12...")
    plot_m12_stacking_weights(meta_learner, stacking_members, trained,
                               val_feats_p, val_labels, best_ml_preds,
                               best_ind_name, val_accs, class_names, PLOTS_ML)

    # -------------------------------------------------------------------------
    # SAVE FINAL RESULTS JSON
    # -------------------------------------------------------------------------
    final = {
        'swin': {
            'val_accuracy': round(float(swin_acc),    4),
            'f1_weighted':  round(float(f1_w),         4),
            'best_epoch':   int(best_epoch),
        },
        'ml_best': {
            'model_name':   best_ml_name,
            'val_accuracy': round(float(best_ml_acc), 4),
            'f1_weighted':  round(float(f1ml_w),       4),
            'delta_%':      round(float((best_ml_acc - swin_acc) * 100), 2),
        },
        'stacking': {
            'val_accuracy': round(float(stack_acc),   4),
            'members':      stacking_members,
        },
        'w_ensemble': {
            'val_accuracy': round(float(w_ens_acc),   4),
        },
        'all_candidates': {
            n: round(float(candidates[n][0]), 4) for n in candidates
        },
    }
    for bname, (bacc, ba, *_) in blend_results.items():
        final[bname] = {
            'val_accuracy': round(float(bacc), 4),
            'alpha':        round(float(ba),   2),
        }

    with open(LOGS_DIR / 'final_results_v3.json', 'w') as f:
        json.dump(final, f, indent=2)

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  PIPELINE COMPLETE")
    print("=" * 72)
    print(f"\n  Swin Transformer   : {swin_acc*100:.2f}%")
    print(f"  Stacking Meta-CLF  : {stack_acc*100:.2f}%  "
          f"(Delta {(stack_acc-swin_acc)*100:+.2f}%)")
    print(f"  Weighted Ensemble  : {w_ens_acc*100:.2f}%  "
          f"(Delta {(w_ens_acc-swin_acc)*100:+.2f}%)")
    for bname, (bacc, ba, *_) in blend_results.items():
        print(f"  {bname:<22}: {bacc*100:.2f}%  "
              f"(alpha={ba:.2f}, Delta {(bacc-swin_acc)*100:+.2f}%)")
    print(f"\n  BEST : {best_ml_name}  ->  {best_ml_acc*100:.2f}%")

    if best_ml_acc >= 0.95:
        print("  TARGET MET - >= 95%")
    elif best_ml_acc >= 0.94:
        print("  TARGET MET - >= 94%")
    elif best_ml_acc >= 0.93:
        print("  CLOSE - >= 93%  (within 1% of target)")
    elif best_ml_acc > swin_acc:
        print(f"  ML BEATS SWIN by {(best_ml_acc-swin_acc)*100:.2f}%")
    else:
        print("  ML did not beat Swin. Try REUSE_FEATURES=False to re-extract features.")

    print(f"\n  Individual results:")
    for n in sorted(val_accs, key=val_accs.get, reverse=True):
        flag = "BEAT" if val_accs[n] > swin_acc else "    "
        print(f"    [{flag}] {n:<18}: {val_accs[n]*100:.2f}%")

    print(f"\n  Swin plots -> {PLOTS_SWIN}/  (S1-S6)")
    print(f"  ML plots   -> {PLOTS_ML}/  (M1-M12)")
    print("=" * 72)