import torch
import torch.nn as nn

try:
    from storage.temporal import TemporalConvNet
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from storage.temporal import TemporalConvNet


DEFAULT_TEMPORAL_LEVELS = 5
MODEL_ARCH_VERSION = "causal-tcn-lstm-multipool-v2"


class TemporalSequenceModel(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, dropout,
                 lstm_hidden, lstm_layers, embedding_dim,
                 use_weight_norm=True):
        super().__init__()
        self.tcn = TemporalConvNet(
            num_inputs=input_channels,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_weight_norm=use_weight_norm,
        )
        self.lstm = nn.LSTM(
            input_size=num_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(3 * lstm_hidden, embedding_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        sequence = self.tcn(inputs).transpose(1, 2)
        sequence, _ = self.lstm(sequence)
        last_feature = sequence[:, -1, :]
        mean_feature = sequence.mean(dim=1)
        max_feature = sequence.amax(dim=1)
        summary = torch.cat(
            [last_feature, mean_feature, max_feature],
            dim=1,
        )
        return self.linear(self.dropout(summary))
