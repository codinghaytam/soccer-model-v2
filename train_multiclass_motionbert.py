from pathlib import Path
import random
import sys
import os

# Set alloc configs to reduce fragmentation (new and old env vars)
os.environ["PYTORCH_ALLOC_CONF"] = "cuda:expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add MotionBERT to Python path
MB_ROOT = Path("MotionBERT")
sys.path.append(str(MB_ROOT))
sys.path.append(str(MB_ROOT / "lib"))
sys.path.append(str(MB_ROOT / "lib" / "model"))

from data_loader import make_dataloader, collate_variable_seq

# Extra imports for reporting
import numpy as np
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# IDE configuration
CSV_PATH = "data/keypoints.csv"
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-6
DEVICE = "cuda"
TEST_SPLIT = 0.2
SEED = 42
LOCAL_MB_CHECKPOINT = Path("models/bert_classifier.bin")
CONFIG_YAML = Path("MotionBERT/configs/action/MB_ft_NTU60_xsub.yaml")
NUM_JOINTS = 17
CHANNELS = 3   # x,y,conf
PERSONS = 1    # M
OUT_DIR = Path("models/motionbert")
# Replicate videos N times (dataset multiplier)
MULTIPLIER = 1

LIGHT_CFG_OVERRIDES = {
    "dim_feat": 128,
    "dim_rep": 256,
    "depth": 3,
    "num_heads": 4,
    "dropout_ratio": 0.0,
}


def _load_mb_config(path: Path):
    cfg = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            try:
                cfg = yaml.safe_load(f) or {}
            except Exception:
                cfg = {}
    # Apply lightweight overrides to reduce memory footprint
    cfg.update({k: LIGHT_CFG_OVERRIDES[k] for k in LIGHT_CFG_OVERRIDES})
    return cfg


def _build_motionbert_backbone(cfg: dict):
    from DSTformer import DSTformer
    # Map YAML keys to DSTformer signature
    dim_in = int(cfg.get('dim_in', 3))
    dim_out = int(cfg.get('dim_out', 3))
    dim_feat = int(cfg.get('dim_feat', 256))
    dim_rep = int(cfg.get('dim_rep', 512))
    depth = int(cfg.get('depth', 5))
    num_heads = int(cfg.get('num_heads', 8))
    mlp_ratio = float(cfg.get('mlp_ratio', 4))
    num_joints = int(cfg.get('num_joints', NUM_JOINTS))
    maxlen = int(cfg.get('maxlen', 243))
    qkv_bias = bool(cfg.get('qkv_bias', True))
    qk_scale = cfg.get('qk_scale', None)
    drop_rate = float(cfg.get('dropout_ratio', 0.0))
    attn_drop_rate = float(cfg.get('attn_drop_rate', 0.0))
    drop_path_rate = float(cfg.get('drop_path_rate', 0.0))
    norm_layer = nn.LayerNorm
    att_fuse = bool(cfg.get('att_fuse', True))

    backbone = DSTformer(
        dim_in=dim_in,
        dim_out=dim_out,
        dim_feat=dim_feat,
        dim_rep=dim_rep,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_joints=num_joints,
        maxlen=maxlen,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
        att_fuse=att_fuse,
    )
    return backbone


def _build_actionnet(backbone, cfg: dict, num_classes: int, num_joints: int):
    from model_action import ActionNet
    dim_rep = int(cfg.get('dim_rep', cfg.get('dim_feat', 512)))
    dropout_ratio = float(cfg.get('dropout_ratio', 0.0))
    hidden_dim = int(cfg.get('hidden_dim', 2048))
    version = 'class'
    model = ActionNet(backbone=backbone, dim_rep=dim_rep, num_classes=num_classes,
                      dropout_ratio=dropout_ratio, version=version, hidden_dim=hidden_dim,
                      num_joints=num_joints)
    return model


def _reshape_to_actionnet_input(x_bt: torch.Tensor, persons: int, joints: int, channels: int) -> torch.Tensor:
    # x_bt: (B, T, F) where F=joints*channels
    B, T, F = x_bt.shape
    expected = joints * channels
    if F != expected:
        raise ValueError(f"Expected feature dim {expected} (J*C), got {F}")
    x = x_bt.view(B, T, joints, channels)  # (B,T,J,C)
    # Duplicate person dimension: if single person data, create second zero person to match M=2 expected
    x1 = x.unsqueeze(1)  # (B,1,T,J,C)
    x2 = torch.zeros_like(x1)
    x_mc = torch.cat([x1, x2], dim=1)  # (B,2,T,J,C)
    if persons == 1:
        return x1
    return x_mc


def _plot_learning_curves(train_losses, val_losses, out_path):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="train")
    if val_losses:
        plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MotionBERT Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_confusion(cm, classes, out_path):
    plt.figure(figsize=(10,8))
    cmap = plt.cm.get_cmap('Greens')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("MotionBERT Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train_multiclass_motionbert(csv_path: str, epochs: int, batch_size: int, lr: float, device: str):
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all sequences (no label filter)
    loader_full, ds = make_dataloader(csv_path, batch_size=batch_size, label_filter=None, shuffle=True)
    if len(ds) == 0:
        print("Dataset is empty. Check CSV path and contents.")
        return

    # Build label mapping across all sequences excluding 'substitution'
    filtered_all = [(x, lbl) for (x, lbl) in ds.sequences if str(lbl).lower() != 'substitution']
    if len(filtered_all) == 0:
        print("No sequences left after filtering 'substitution'.")
        return
    # Duplicate dataset N times according to MULTIPLIER
    if MULTIPLIER > 1:
        augmented_all = filtered_all * int(MULTIPLIER)
    else:
        augmented_all = filtered_all
    labels = sorted({lbl for _, lbl in filtered_all})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    num_classes = len(label_to_idx)

    # Load MB config and build backbone + ActionNet
    cfg = _load_mb_config(CONFIG_YAML)
    backbone = _build_motionbert_backbone(cfg)
    model = _build_actionnet(backbone, cfg, num_classes=num_classes, num_joints=NUM_JOINTS).to(device_t)

    # Load checkpoint strictly
    if not LOCAL_MB_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found at {LOCAL_MB_CHECKPOINT}")
    sd = torch.load(str(LOCAL_MB_CHECKPOINT), map_location='cpu')
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    try:
        model.load_state_dict(sd, strict=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load state_dict: {e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device_t.type == "cuda"))

    # Train/Val split
    sequences = list(augmented_all)
    random.seed(SEED)
    random.shuffle(sequences)
    n_test = max(1, int(len(sequences) * TEST_SPLIT)) if len(sequences) > 1 else 0
    val_seqs = sequences[:n_test] if n_test > 0 else []
    train_seqs = sequences[n_test:] if n_test > 0 else sequences

    def _seqs_to_loader(seqs, shuffle, bs: int | None = None):
        class _SeqDS(torch.utils.data.Dataset):
            def __init__(self, seqs, label_to_idx):
                self.seqs = seqs
                self.label_to_idx = label_to_idx
            def __len__(self):
                return len(self.seqs)
            def __getitem__(self, i):
                x, label = self.seqs[i]
                return x, self.label_to_idx[label]
        ds_local = _SeqDS(seqs, label_to_idx)
        used_bs = bs if bs is not None else batch_size
        return DataLoader(ds_local, batch_size=used_bs, shuffle=shuffle, collate_fn=collate_variable_seq)

    train_loader = _seqs_to_loader(train_seqs, True)
    val_loader = _seqs_to_loader(val_seqs, False) if val_seqs else None

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0
        correct = 0
        for x_bt, lengths, y in train_loader:
            # x_bt: (B,T,F) -> (B,M,T,J,C)
            x_bt = x_bt.to(device_t)
            y = y.to(device_t)
            x_in = _reshape_to_actionnet_input(x_bt, PERSONS, NUM_JOINTS, CHANNELS)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device_t.type == "cuda")):
                logits = model(x_in)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x_bt.size(0)
            n_samples += x_bt.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).float().sum().item()
        train_loss = total_loss / max(n_samples, 1)
        train_acc = correct / max(n_samples, 1)
        train_losses.append(train_loss)

        val_msg = ""
        v_loss_epoch = None
        if val_loader is not None:
            model.eval()
            v_total = 0.0
            v_count = 0
            v_correct = 0
            with torch.no_grad():
                for x_bt, lengths, y in val_loader:
                    x_bt = x_bt.to(device_t)
                    y = y.to(device_t)
                    x_in = _reshape_to_actionnet_input(x_bt, PERSONS, NUM_JOINTS, CHANNELS)
                    with torch.cuda.amp.autocast(enabled=(device_t.type == "cuda")):
                        logits = model(x_in)
                        loss = criterion(logits, y)
                    v_total += loss.item() * x_bt.size(0)
                    v_count += x_bt.size(0)
                    preds = logits.argmax(dim=1)
                    v_correct += (preds == y).float().sum().item()
            val_loss = v_total / max(v_count, 1)
            val_acc = v_correct / max(v_count, 1)
            v_loss_epoch = val_loss
            val_msg = f" | val loss {val_loss:.4f} acc {val_acc:.3f}"
        val_losses.append(v_loss_epoch if v_loss_epoch is not None else np.nan)

        print(f"[MB-MC] Epoch {epoch}/{epochs} - loss {train_loss:.4f} acc {train_acc:.3f}{val_msg}")

    # Save final classifier head weights (if you want full model, save entire state)
    out_dir = Path("models/motionbert")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "motionbert_multiclass.pt"
    torch.save({
        "model_state": model.state_dict(),
        "label_to_idx": label_to_idx,
        "config": cfg,
        "persons": PERSONS,
        "joints": NUM_JOINTS,
        "channels": CHANNELS,
    }, ckpt_path)
    print(f"Saved multiclass MotionBERT-based model to {ckpt_path}")

    # Learning curves
    _plot_learning_curves(train_losses, val_losses, OUT_DIR / "learning_curve.png")

    # Confusion matrices and sample predictions on train and val
    def _collect_preds(loader):
        ys_true, ys_pred = [], []
        model.eval()
        with torch.no_grad():
            for x_bt, lengths, y in loader:
                x_bt = x_bt.to(device_t)
                y = y.to(device_t)
                x_in = _reshape_to_actionnet_input(x_bt, PERSONS, NUM_JOINTS, CHANNELS)
                logits = model(x_in)
                preds = logits.argmax(dim=1)
                ys_true.extend(y.cpu().numpy().tolist())
                ys_pred.extend(preds.cpu().numpy().tolist())
        return np.array(ys_true), np.array(ys_pred)

    classes = [labels[i] for i in range(num_classes)]
    if len(train_seqs) > 0:
        tr_loader = _seqs_to_loader(train_seqs, False)
        y_true_tr, y_pred_tr = _collect_preds(tr_loader)
        cm_tr = confusion_matrix(y_true_tr, y_pred_tr, labels=list(range(num_classes)))
        _plot_confusion(cm_tr, classes, OUT_DIR / "confusion_train.png")
        rep_tr = classification_report(y_true_tr, y_pred_tr, target_names=classes, digits=3)
        (OUT_DIR / "report_train.txt").write_text(rep_tr, encoding="utf-8")

    if val_loader is not None:
        y_true_val, y_pred_val = _collect_preds(val_loader)
        cm_val = confusion_matrix(y_true_val, y_pred_val, labels=list(range(num_classes)))
        _plot_confusion(cm_val, classes, OUT_DIR / "confusion_val.png")
        rep_val = classification_report(y_true_val, y_pred_val, target_names=classes, digits=3)
        (OUT_DIR / "report_val.txt").write_text(rep_val, encoding="utf-8")

    # Sample predictions
    samples_path = OUT_DIR / "samples.txt"
    with torch.no_grad():
        lines = []
        sample_loader = _seqs_to_loader((val_seqs or train_seqs)[:min(16, len(train_seqs))], False, bs=1)
        for x_bt, lengths, y in sample_loader:
            x_in = _reshape_to_actionnet_input(x_bt.to(device_t), PERSONS, NUM_JOINTS, CHANNELS)
            logits = model(x_in)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(probs.argmax())
            top5_idx = np.argsort(probs)[-5:][::-1]
            top5 = [(classes[i], float(probs[i])) for i in top5_idx]
            lines.append(f"true={classes[int(y.item())]} pred={classes[pred]} top5={top5}")
        samples_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    random.seed(SEED)
    train_multiclass_motionbert(CSV_PATH, EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE)
