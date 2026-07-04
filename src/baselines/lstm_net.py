"""
Minimal LSTM network used by the directional deep-learning baseline.

The architecture follows the standard single-layer LSTM directional classifier
of Fischer and Krauss (2018): a recurrent layer over a short window of
multi-channel features, followed by dropout and a single linear unit producing
the logit for the "next-day up" class.  Kept intentionally small to remain a
fair, lightweight reference point rather than a heavily tuned competitor.
"""
from __future__ import annotations

import torch
from torch import nn


class DirectionalLSTMNet(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 48,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) -> logits: (batch,)
        outputs, _ = self.lstm(x)
        last_step = outputs[:, -1, :]
        logits = self.head(self.dropout(last_step))
        return logits.squeeze(-1)
