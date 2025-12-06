import torch
import torch.nn as nn


class LSTMScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # regression score

    def forward(self, x, lengths=None):
        # x: (B, T, F). lengths optional for variable length (not strictly required here)
        packed_out, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])  # last layer hidden state
        return out.squeeze(-1)


class CNN1DScorer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x, lengths=None):
        # x: (B, T, F) where F=input_dim
        x = x.transpose(1, 2)  # -> (B, F, T)
        x = self.conv(x).squeeze(-1)
        out = self.fc(x)
        return out.squeeze(-1)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths=None):
        # x: (B, T, F)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])  # (B, num_classes)
        return out


class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, lengths=None):
        # x: (B, T, F) where F=input_dim
        x = x.transpose(1, 2)  # -> (B, F, T)
        x = self.conv(x).squeeze(-1)
        logits = self.fc(x)
        return logits
