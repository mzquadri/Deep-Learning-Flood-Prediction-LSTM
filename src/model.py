"""
LSTM model for river discharge (flood) prediction.

Architecture:
- Multi-layer LSTM encoder
- Fully connected decoder head
- Optional dropout for regularization
"""

import torch
import torch.nn as nn


class FloodLSTM(nn.Module):
    """
    LSTM-based model for one-step-ahead river discharge prediction.

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    hidden_size : int
        Number of LSTM hidden units.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability (applied between LSTM layers).
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_size)

        Returns
        -------
        torch.Tensor, shape (batch,)
            Predicted discharge at the next time step.
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Use the last time step's hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)

        # Decode to prediction
        out = self.fc(last_hidden).squeeze(-1)  # (batch,)
        return out

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = FloodLSTM(input_size=3, hidden_size=128, num_layers=2, dropout=0.2)
    print(f"Model architecture:\n{model}")
    print(f"\nTrainable parameters: {model.count_parameters():,}")

    # Test forward pass
    dummy = torch.randn(16, 30, 3)  # batch=16, seq_len=30, features=3
    output = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")
