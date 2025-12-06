from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import make_dataloader, collate_variable_seq
from models.models import LSTMClassifier

# IDE configuration
CSV_PATH = "data/keypoints.csv"
EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
DEVICE = "cuda"  # or "cpu"
TEST_SPLIT = 0.2
SEED = 42


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
            return x, self.label_to_idx[label]

    labels = sorted({lbl for _, lbl in sequences})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    ds = _SeqDataset(sequences, label_to_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_variable_seq), label_to_idx


def train_multiclass(csv_path: str, epochs: int, batch_size: int, lr: float, device: str):
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load all sequences (no label filter)
    loader_full, ds = make_dataloader(csv_path, batch_size=batch_size, label_filter=None, shuffle=True)
    if len(ds) == 0:
        print("Dataset is empty. Check CSV path and contents.")
        return

    # Train/val split across all sequences
    train_seqs, val_seqs = _split_sequences(ds.sequences, TEST_SPLIT)
    if len(train_seqs) == 0:
        print("Not enough sequences to train.")
        return

    # Build loaders
    train_loader, label_to_idx = _make_loader_from_sequences(train_seqs, batch_size, shuffle=True)
    val_loader, _ = _make_loader_from_sequences(val_seqs, batch_size, shuffle=False) if val_seqs else (None, None)
    num_classes = len(label_to_idx)

    # Infer input_dim
    sample_x, _ = train_seqs[0]
    input_dim = sample_x.shape[1]

    # Model and training setup
    model = LSTMClassifier(input_dim=input_dim, num_classes=num_classes).to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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

        # Validate
        val_msg = ""
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
            val_msg = f" | val loss {val_loss:.4f} acc {val_acc:.3f}"

        print(f"[LSTM-MC] Epoch {epoch}/{epochs} - loss {train_loss:.4f} acc {train_acc:.3f}{val_msg}")

    # Save single multiclass model
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "lstm_multiclass.pt"
    torch.save({"model_state": model.state_dict(), "input_dim": input_dim, "num_classes": num_classes,
                "label_to_idx": label_to_idx}, ckpt_path)
    print(f"Saved multiclass LSTM model to {ckpt_path}")


if __name__ == "__main__":
    random.seed(SEED)
    train_multiclass(CSV_PATH, EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE)
