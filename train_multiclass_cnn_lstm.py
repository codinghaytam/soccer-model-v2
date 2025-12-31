from pathlib import Path
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import make_dataloader, collate_variable_seq

# Evaluation and plotting
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IDE configuration (no CLI)
CSV_PATH = "data/keypoints.csv"
EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
DEVICE = "cuda"  # or "cpu"
TEST_SPLIT = 0.2
SEED = 42
OUT_DIR = Path("models/cnn_lstm")
# Replicate videos N times (dataset multiplier)
MULTIPLIER = 5
EARLY_STOP_ACC = 0.98  # stop when val accuracy >= this

# Rotation augmentation angles (radians) per part
ROT_ANGLE_ARMS = 0.5
ROT_ANGLE_LEGS = 0.5
ROT_ANGLE_NECK = 0.5
ROT_ANGLE_BODY = 0.5

# YOLO-Pose joint mapping indices
JOINTS = {
    "nose": 0,
    "l_eye": 1,
    "r_eye": 2,
    "l_ear": 3,
    "r_ear": 4,
    "l_sho": 5,
    "r_sho": 6,
    "l_elb": 7,
    "r_elb": 8,
    "l_wri": 9,
    "r_wri": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_kne": 13,
    "r_kne": 14,
    "l_ank": 15,
    "r_ank": 16,
}

ARMS = [JOINTS["l_sho"], JOINTS["r_sho"], JOINTS["l_elb"], JOINTS["r_elb"], JOINTS["l_wri"], JOINTS["r_wri"]]
LEGS = [JOINTS["l_hip"], JOINTS["r_hip"], JOINTS["l_kne"], JOINTS["r_kne"], JOINTS["l_ank"], JOINTS["r_ank"]]
NECK = [JOINTS["nose"], JOINTS["l_eye"], JOINTS["r_eye"], JOINTS["l_ear"], JOINTS["r_ear"]]
BODY = [JOINTS["l_sho"], JOINTS["r_sho"], JOINTS["l_hip"], JOINTS["r_hip"]]


def _split_sequences(ds_sequences, test_split):
    seqs = list(ds_sequences)
    random.shuffle(seqs)
    n_test = max(1, int(len(seqs) * test_split)) if len(seqs) > 1 else 0
    test_seqs = seqs[:n_test] if n_test > 0 else []
    train_seqs = seqs[n_test:] if n_test > 0 else seqs
    return train_seqs, test_seqs


def _make_loader_from_sequences(sequences, batch_size, shuffle=True):
    class _SeqDataset(torch.utils.data.Dataset):
        def __init__(self, seqs, label_to_idx):
            self.seqs = seqs
            self.label_to_idx = label_to_idx
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, i):
            x, label = self.seqs[i]
            # x: (T, D) where D = J*2 (+scores if present). Apply augmentation per sequence.
            x_aug = _rotate_parts_in_seq(x)
            return x_aug, self.label_to_idx[label]

    labels = sorted({lbl for _, lbl in sequences})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    ds = _SeqDataset(sequences, label_to_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_variable_seq), label_to_idx


def _plot_learning_curves(train_losses, val_losses, out_path, title):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="train")
    if val_losses:
        plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_confusion(cm, classes, out_path, title):
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


class CNNLSTMClassifier(nn.Module):
    """A simple 1D CNN over time followed by LSTM for sequence classification.
    Input: (batch, T, D)
    """
    def __init__(self, input_dim: int, num_classes: int, cnn_channels: int = 128, lstm_hidden: int = 256, lstm_layers: int = 1, dropout: float = 0.3, weight_decay: float = 1e-4):
        super().__init__()
        self.input_dim = input_dim
        # CNN expects (batch, C, T). Use C=input_dim.
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden*2, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden, num_classes)
        )
        # Regularization parameter to be used by optimizer
        self.weight_decay = weight_decay

    def forward(self, x):
        # x: (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        feat = self.cnn(x)     # (B, C', T)
        feat = feat.transpose(1, 2)  # (B, T, C')
        out, (h, c) = self.lstm(feat)
        # Use last time step or mean pool
        pooled = out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


def _rotate2d(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 2D points by angle around origin.
    points: (..., 2)
    """
    if angle == 0.0:
        return points
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return points @ rot.T


def _rotate_parts_in_seq(x: torch.Tensor) -> torch.Tensor:
    """Rotate selected joint groups within a single sequence tensor.
    x: (T, D). D is expected to be J*2 optionally +scores.
    We only rotate the XY coordinates of the first J*2 columns; any extra columns (e.g., scores) remain unchanged.
    """
    # Convert to numpy for easier slicing
    x_np = x.detach().cpu().numpy()
    T, D = x_np.shape
    J = 17
    xy_dim = J * 2
    if D < xy_dim:
        return x  # nothing to do
    coords = x_np[:, :xy_dim].reshape(T, J, 2)
    # Compute centers for body rotation (mean over BODY joints)
    body_center = coords[:, BODY, :].mean(axis=1, keepdims=True)  # (T, 1, 2)
    # Apply rotations
    def apply_group(group: List[int], angle: float, center: np.ndarray | None):
        if angle == 0.0:
            return
        pts = coords[:, group, :]
        if center is not None:
            pts_rel = pts - center
            pts_rot = _rotate2d(pts_rel.reshape(-1, 2), angle).reshape(pts_rel.shape)
            coords[:, group, :] = pts_rot + center
        else:
            pts_rot = _rotate2d(pts.reshape(-1, 2), angle).reshape(pts.shape)
            coords[:, group, :] = pts_rot
    apply_group(ARMS, ROT_ANGLE_ARMS, body_center)
    apply_group(LEGS, ROT_ANGLE_LEGS, body_center)
    apply_group(NECK, ROT_ANGLE_NECK, body_center)
    apply_group(BODY, ROT_ANGLE_BODY, body_center)

    x_np[:, :xy_dim] = coords.reshape(T, xy_dim)
    # Return torch tensor with same dtype/device
    x_aug = torch.from_numpy(x_np).to(x.device, dtype=x.dtype)
    return x_aug


def train_multiclass(csv_path: str, epochs: int, batch_size: int, lr: float, device: str):
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all sequences
    loader_full, ds = make_dataloader(csv_path, batch_size=batch_size, label_filter=None, shuffle=True)
    if len(ds) == 0:
        print("Dataset is empty. Check CSV path and contents.")
        return

    # Filter out 'substitution'
    filtered = [(x, lbl) for (x, lbl) in ds.sequences if str(lbl).lower() != 'substitution']
    if len(filtered) == 0:
        print("No sequences left after filtering 'substitution'.")
        return

    # Replicate videos N times
    if MULTIPLIER > 1:
        filtered = filtered * int(MULTIPLIER)

    # Split
    train_seqs, val_seqs = _split_sequences(filtered, TEST_SPLIT)
    if len(train_seqs) == 0:
        print("Not enough sequences to train.")
        return

    # Build loaders
    train_loader, label_to_idx = _make_loader_from_sequences(train_seqs, batch_size, shuffle=True)
    val_loader, _ = _make_loader_from_sequences(val_seqs, batch_size, shuffle=False) if val_seqs else (None, None)
    num_classes = len(label_to_idx)
    idx_to_label = {v:k for k,v in label_to_idx.items()}

    # Infer input_dim
    sample_x, _ = train_seqs[0]
    input_dim = sample_x.shape[1]

    # Model
    model = CNNLSTMClassifier(input_dim=input_dim, num_classes=num_classes)
    model = model.to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=model.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        n_samples = 0
        correct = 0
        with torch.enable_grad():
            for x, lengths, y in train_loader:
                x = x.to(device_t)
                y = y.to(device_t)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                n_samples += x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
        train_loss = total_loss / max(n_samples, 1)
        train_acc = correct / max(n_samples, 1)
        train_losses.append(train_loss)

        # Validate
        val_msg = ""
        v_loss_epoch = None
        if val_loader is not None:
            model.eval()
            v_total = 0.0
            v_count = 0
            v_correct = 0
            with torch.no_grad():
                for x, lengths, y in val_loader:
                    x = x.to(device_t)
                    y = y.to(device_t)
                    logits = model(x)
                    loss = criterion(logits, y)
                    v_total += loss.item() * x.size(0)
                    v_count += x.size(0)
                    preds = logits.argmax(dim=1)
                    v_correct += (preds == y).sum().item()
            val_loss = v_total / max(v_count, 1)
            val_acc = v_correct / max(v_count, 1)
            v_loss_epoch = val_loss
            val_msg = f" | val loss {val_loss:.4f} acc {val_acc:.3f}"
        val_losses.append(v_loss_epoch if v_loss_epoch is not None else np.nan)

        print(f"[CNN-LSTM] Epoch {epoch}/{epochs} - loss {train_loss:.4f} acc {train_acc:.3f}{val_msg}")

        # Track best and save
        if val_loader is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = OUT_DIR / "cnn_lstm_multiclass.pt"
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "num_classes": num_classes,
                "label_to_idx": label_to_idx
            }, ckpt_path)
            print(f"Saved CNN+LSTM multiclass model to {ckpt_path}")

        # Early stopping
        if val_loader is not None and val_acc >= EARLY_STOP_ACC:
            print(f"[CNN-LSTM] Early stop at epoch {epoch} with val acc {val_acc:.3f}")
            break

    # Learning curves
    _plot_learning_curves(train_losses, val_losses, OUT_DIR / "learning_curve.png", title="CNN+LSTM Learning Curves")

    # Confusion matrices and reports
    def _collect_preds(loader):
        ys_true, ys_pred = [], []
        model.eval()
        with torch.no_grad():
            for x, lengths, y in loader:
                x = x.to(device_t)
                y = y.to(device_t)
                logits = model(x)
                preds = logits.argmax(dim=1)
                ys_true.extend(y.cpu().numpy().tolist())
                ys_pred.extend(preds.cpu().numpy().tolist())
        return np.array(ys_true), np.array(ys_pred)

    if len(train_seqs) > 0:
        tr_loader, _ = _make_loader_from_sequences(train_seqs, batch_size, shuffle=False)
        y_true_tr, y_pred_tr = _collect_preds(tr_loader)
        classes = [idx_to_label[i] for i in range(num_classes)]
        cm_tr = confusion_matrix(y_true_tr, y_pred_tr, labels=list(range(num_classes)))
        _plot_confusion(cm_tr, classes, OUT_DIR / "confusion_train.png", title="CNN+LSTM Confusion (Train)")
        rep_tr = classification_report(y_true_tr, y_pred_tr, target_names=classes, digits=3)
        (OUT_DIR / "report_train.txt").write_text(rep_tr, encoding="utf-8")

    if val_loader is not None:
        y_true_val, y_pred_val = _collect_preds(val_loader)
        classes = [idx_to_label[i] for i in range(num_classes)]
        cm_val = confusion_matrix(y_true_val, y_pred_val, labels=list(range(num_classes)))
        _plot_confusion(cm_val, classes, OUT_DIR / "confusion_val.png", title="CNN+LSTM Confusion (Val)")
        rep_val = classification_report(y_true_val, y_pred_val, target_names=classes, digits=3)
        (OUT_DIR / "report_val.txt").write_text(rep_val, encoding="utf-8")

    # Sample predictions
    samples_path = OUT_DIR / "samples.txt"
    with torch.no_grad():
        lines = []
        # ensure classes mapping is available
        classes = [idx_to_label[i] for i in range(num_classes)]
        sample_loader, _ = _make_loader_from_sequences((val_seqs or train_seqs)[:min(16, len(train_seqs))], batch_size=1, shuffle=False)
        for x, lengths, y in sample_loader:
            logits = model(x.to(device_t))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(probs.argmax())
            true_idx = int(y.item())
            true_label = classes[true_idx]
            pred_label = classes[pred_idx]
            pred_prob = float(probs[pred_idx])
            top5_idx = np.argsort(probs)[-5:][::-1]
            top5 = [(classes[i], float(probs[i])) for i in top5_idx]
            lines.append(f"true={true_label} pred={pred_label} prob={pred_prob:.4f} top5={top5}")
        samples_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    random.seed(SEED)
    train_multiclass(CSV_PATH, EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE)
