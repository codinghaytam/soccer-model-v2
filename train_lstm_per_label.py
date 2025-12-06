from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random

from data_loader import make_dataloader, get_all_labels, collate_variable_seq
from models.models import LSTMScorer


# Configuration for IDE usage (no CLI)
CSV_PATH = "data/keypoints.csv"
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
DEVICE = "cuda"  # or "cpu"
TEST_SPLIT = 0.2  # fraction of sequences reserved for validation
SEED = 42


def _make_loaders_from_sequences(sequences, batch_size, shuffle=True):
    class _SeqDataset(torch.utils.data.Dataset):
        def __init__(self, seqs, label_to_idx):
            self.seqs = seqs
            self.label_to_idx = label_to_idx
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, i):
            x, label = self.seqs[i]
            return x, self.label_to_idx[label]
    # build label_to_idx from provided sequences
    labels = sorted({lbl for _, lbl in sequences})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    ds = _SeqDataset(sequences, label_to_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_variable_seq)


def train_all_labels(csv_path: str, epochs: int, batch_size: int, lr: float, device: str):
    labels = get_all_labels(csv_path)
    if not labels:
        print("No labels found in CSV. Nothing to train.")
        return

    random.seed(SEED)
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    for label in labels:
        # Build dataset for this label
        loader_full, ds = make_dataloader(csv_path, batch_size=batch_size, label_filter=label, shuffle=True)
        if len(ds) == 0:
            print(f"Skipping label '{label}': no sequences.")
            continue

        # Train/test split on sequences
        seqs = list(ds.sequences)
        random.shuffle(seqs)
        n_test = max(1, int(len(seqs) * TEST_SPLIT)) if len(seqs) > 1 else 0
        test_seqs = seqs[:n_test] if n_test > 0 else []
        train_seqs = seqs[n_test:] if n_test > 0 else seqs
        if len(train_seqs) == 0:
            print(f"Skipping label '{label}': not enough sequences for training.")
            continue

        train_loader = _make_loaders_from_sequences(train_seqs, batch_size, shuffle=True)
        val_loader = _make_loaders_from_sequences(test_seqs, batch_size, shuffle=False) if test_seqs else None

        # Infer input_dim from first training sequence
        sample_x, _ = train_seqs[0]
        input_dim = sample_x.shape[1]

        model = LSTMScorer(input_dim=input_dim).to(device_t)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            total_loss = 0.0
            n_samples = 0
            for x, lengths, y in train_loader:
                target = torch.ones(x.size(0), dtype=torch.float32)  # TODO: replace with real target
                x = x.to(device_t)
                target = target.to(device_t)
                optimizer.zero_grad()
                preds = model(x)
                loss = criterion(preds, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                n_samples += x.size(0)
            train_loss = total_loss / max(n_samples, 1)

            # Validate
            val_loss = None
            if val_loader is not None:
                model.eval()
                v_total = 0.0
                v_count = 0
                with torch.no_grad():
                    for x, lengths, y in val_loader:
                        target = torch.ones(x.size(0), dtype=torch.float32)
                        x = x.to(device_t)
                        target = target.to(device_t)
                        preds = model(x)
                        loss = criterion(preds, target)
                        v_total += loss.item() * x.size(0)
                        v_count += x.size(0)
                val_loss = v_total / max(v_count, 1)

            msg = f"[Label {label}] Epoch {epoch}/{epochs} - train {train_loss:.4f}"
            if val_loss is not None:
                msg += f" | val {val_loss:.4f}"
            print(msg)

        out_dir = Path("models")
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"lstm_scorer_label_{label}.pt"
        torch.save({"model_state": model.state_dict(), "input_dim": input_dim}, ckpt_path)
        print(f"Saved model for label '{label}' to {ckpt_path}")


if __name__ == "__main__":
    # Train for all labels found in the CSV with train/validation split
    train_all_labels(CSV_PATH, EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE)
