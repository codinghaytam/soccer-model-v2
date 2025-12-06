## LSTM
````python
import torch
import torch.nn as nn

class LSTMScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Output: score

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])  # Last layer's hidden state
        return out.squeeze(-1)

# Usage Example
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMScorer(input_dim=34).to(device)  # Example: 17 joints x 2D = 34 features
# Continue with your DataLoader and optimizer...

````
## ST-GCN (Spatial-Temporal Graph Convolutional Network)
````python
# Install PYSKL/MMAction2 first (see docs)
# https://github.com/kennymckormick/pyskl
# Sample inference
from mmcv import Config
from pyskl.apis import inference_recognizer, init_recognizer

config = 'configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py'
checkpoint = 'checkpoints/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.pth'
model = init_recognizer(config, checkpoint, device='cuda')
# Prepare your 2D skeleton JSON or numpy input
score = inference_recognizer(model, 'your_skeleton_file.json')
print('Score:', score)

````
## 1D CNN Model (for sequential features)
````python
import torch
import torch.nn as nn

class CNN1DScorer(nn.Module):
    def __init__(self, n_joints=17, channels=2, seq_len=60):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_joints*channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, n_joints * channels)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.conv(x).squeeze(-1)
        out = self.fc(x)
        return out.squeeze(-1)

# Usage Example
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN1DScorer(n_joints=17, channels=2, seq_len=60).to(device)
# Continue with your DataLoader and optimizer...

````