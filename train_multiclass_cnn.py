from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import make_dataloader, collate_variable_seq
from models.models import CNN1DClassifier

# Extra imports for reporting
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IDE configuration
CSV_PATH = "data/keypoints.csv"
EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 1e-2
DEVICE = "cuda"  # or "cpu"
TEST_SPLIT = 0.2
SEED = 42
OUT_DIR = Path("models/cnn")
# Replicate videos N times (dataset multiplier)
MULTIPLIER = 1


def _split_sequences(ds_sequences, test_split):
    seqs = list(ds_sequences)
    random.shuffle(seqs)
    n_test = max(1, int(len(seqs) * test_split)) if len(seqs) > 1 else 0
    test_seqs = seqs[:n_test] if n_test > 0 else []
    train_seqs = seqs[n_test:] if n_test > 0 else seqs
    return train_seqs, test_seqs


def _make_loader_from_sequences(sequences, batch_size, shuffle=True, label_to_idx=None):
    class _SeqDataset(torch.utils.data.Dataset):
        def __init__(self, seqs, label_to_idx):
            self.seqs = seqs
            self.label_to_idx = label_to_idx
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, i):
            x, label = self.seqs[i]
            return x, self.label_to_idx[label]

    if label_to_idx is None:
        labels = sorted({lbl for _, lbl in sequences})
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    ds = _SeqDataset(sequences, label_to_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_variable_seq), label_to_idx


def _plot_learning_curves(train_losses, val_losses, out_path):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="train")
    if val_losses:
        plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN1D Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_confusion(cm, classes, out_path):
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title("CNN1D Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train_multiclass_cnn(csv_path: str, epochs: int, batch_size: int, lr: float, device: str):
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all sequences (no label filter)
    loader_full, ds = make_dataloader(csv_path, batch_size=batch_size, label_filter=None, shuffle=True)
    if len(ds) == 0:
        print("Dataset is empty. Check CSV path and contents.")
        return

    # Filter out 'substitution' label sequences globally
    filtered = [(x, lbl) for (x, lbl) in ds.sequences if str(lbl).lower() != 'substitution']
    if len(filtered) == 0:
        print("No sequences left after filtering 'substitution'.")
        return

    # Replicate videos N times
    if MULTIPLIER > 1:
        filtered = filtered * int(MULTIPLIER)

    # Build a single label mapping from filtered data (used for both train and val)
    all_labels = sorted({lbl for _, lbl in filtered})
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)

    # Train/val split across filtered sequences
    train_seqs, val_seqs = _split_sequences(filtered, TEST_SPLIT)
    if len(train_seqs) == 0:
        print("Not enough sequences to train.")
        return

    # Build loaders using the same mapping
    train_loader, _ = _make_loader_from_sequences(train_seqs, batch_size, shuffle=True, label_to_idx=label_to_idx)
    val_loader, _ = _make_loader_from_sequences(val_seqs, batch_size, shuffle=False, label_to_idx=label_to_idx) if val_seqs else (None, None)

    # Infer input_dim
    sample_x, _ = train_seqs[0]
    input_dim = sample_x.shape[1]

    # Model and training setup
    model = CNN1DClassifier(input_dim=input_dim, num_classes=num_classes).to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

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

        print(f"[CNN-MC] Epoch {epoch}/{epochs} - loss {train_loss:.4f} acc {train_acc:.3f}{val_msg}")

    # Save single multiclass CNN model
    out_dir = Path("models/cnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "cnn1d_multiclass.pt"
    torch.save({"model_state": model.state_dict(), "input_dim": input_dim, "num_classes": num_classes,
                "label_to_idx": label_to_idx}, ckpt_path)
    print(f"Saved multiclass CNN1D model to {ckpt_path}")

    # Learning curves
    _plot_learning_curves(train_losses, val_losses, OUT_DIR / "learning_curve.png")

    # Confusion matrices and sample predictions on train and val
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

    classes = [idx_to_label[i] for i in range(num_classes)]
    if len(train_seqs) > 0:
        tr_loader, _ = _make_loader_from_sequences(train_seqs, batch_size, shuffle=False, label_to_idx=label_to_idx)
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

    # Sample predictions (omit true labels entirely)
    samples_path = OUT_DIR / "samples.txt"
    with torch.no_grad():
        lines = []
        sample_loader, _ = _make_loader_from_sequences((val_seqs or train_seqs)[:min(16, len(train_seqs))], batch_size=1, shuffle=False, label_to_idx=label_to_idx)
        for x, lengths, _ in sample_loader:
            logits = model(x.to(device_t))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = probs.argmax()
            top5_idx = np.argsort(probs)[-5:][::-1]
            top5 = [(idx_to_label[i], float(probs[i])) for i in top5_idx]
            lines.append(f"pred={idx_to_label[pred]} top5={top5}")
        samples_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    random.seed(SEED)
    train_multiclass_cnn(CSV_PATH, EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE)
