import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class KeypointsDataset(Dataset):
    """Dataset of per-frame keypoints grouped by video and label.

    Tries to auto-detect these columns:
      - video_id (alias: video, vid, file, filename)
      - frame (alias: frame_idx, idx)
      - label (alias: class, movement)
    All other numeric columns are treated as features.
    """

    def __init__(self, csv_path: str, label_filter: str | None = None):
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}

        # Detect label column
        label_col = None
        for cand in ("label", "class", "movement"):
            if cand in cols:
                label_col = cols[cand]
                break
        if label_col is None:
            raise KeyError(f"No label column found. Available columns: {list(df.columns)}")

        # Optional: keep only one label
        if label_filter is not None:
            df = df[df[label_col] == label_filter].copy()

        # Detect video id and frame columns
        video_col = None
        for cand in ("video_id", "video", "vid", "file", "filename"):
            if cand in cols:
                video_col = cols[cand]
                break
        frame_col = None
        for cand in ("frame", "frame_idx", "idx"):
            if cand in cols:
                frame_col = cols[cand]
                break

        # Sort when possible
        sort_cols = []
        if video_col is not None:
            sort_cols.append(video_col)
        if frame_col is not None:
            sort_cols.append(frame_col)
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

        # Numeric feature columns only, excluding meta
        meta_cols = {c for c in [video_col, frame_col, label_col] if c is not None}
        numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in meta_cols]
        if not numeric_cols:
            raise ValueError("No numeric feature columns found in CSV after excluding meta columns.")

        # Build sequences
        self.sequences = []  # list of (tensor[T, F], label)
        if video_col is not None:
            # Group by video and label
            for (vid, label), g in df.groupby([video_col, label_col]):
                feat_df = g[numeric_cols]
                if feat_df.empty:
                    continue
                x = torch.tensor(feat_df.values, dtype=torch.float32)  # (T, F)
                self.sequences.append((x, label))
        else:
            # No video id: treat each row as its own sequence (T=1)
            for _, row in df.iterrows():
                label = row[label_col]
                feat = torch.tensor(row[numeric_cols].values, dtype=torch.float32).unsqueeze(0)
                self.sequences.append((feat, label))

        if not self.sequences:
            raise ValueError(
                f"No sequences built from CSV. Records={len(df)}; labels present={df[label_col].nunique()} "
                f"numeric_features={len(numeric_cols)}; video_col={'yes' if video_col else 'no'}"
            )

        # Build a mapping label -> int index
        labels = sorted({lbl for _, lbl in self.sequences})
        self.label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, label = self.sequences[idx]
        return x, self.label_to_idx[label]


def collate_variable_seq(batch):
    """Pad variable-length sequences in a batch.

    batch: list of (T_i, F) tensors
    returns: padded (B, T_max, F), lengths
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    T_max = int(lengths.max())
    F = xs[0].shape[1]

    padded = torch.zeros(len(xs), T_max, F, dtype=torch.float32)
    for i, x in enumerate(xs):
        padded[i, : x.shape[0]] = x

    ys = torch.tensor(ys, dtype=torch.long)
    return padded, lengths, ys


def make_dataloader(csv_path: str, batch_size: int = 8, label_filter: str | None = None, shuffle: bool = True):
    ds = KeypointsDataset(csv_path, label_filter=label_filter)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_variable_seq,
    )
    return loader, ds


def get_all_labels(csv_path: str):
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    label_col = None
    for cand in ("label", "class", "movement"):
        if cand in cols:
            label_col = cols[cand]
            break
    if label_col is None:
        raise KeyError(f"No label column found. Available columns: {list(df.columns)}")
    return sorted(df[label_col].dropna().unique().tolist())
