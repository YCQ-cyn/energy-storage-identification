import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, dropout=0.2, use_weight_norm=True):
        super().__init__()
        self.left_padding = int(padding)

        conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )
        conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )
        self._init_conv(conv1)
        self._init_conv(conv2)
        if use_weight_norm:
            conv1 = weight_norm(conv1)
            conv2 = weight_norm(conv2)

        self.conv1 = conv1
        self.conv2 = conv2
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs
            else None
        )
        self.relu = nn.ReLU()
        if self.downsample is not None:
            self._init_conv(self.downsample)

    @staticmethod
    def _init_conv(module):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def _causal_conv(self, inputs, convolution):
        if self.left_padding > 0:
            inputs = F.pad(inputs, (self.left_padding, 0))
        return convolution(inputs)

    def forward(self, inputs):
        output = self._causal_conv(inputs, self.conv1)
        output = self.dropout1(self.relu1(output))
        output = self._causal_conv(output, self.conv2)
        output = self.dropout2(self.relu2(output))
        residual = inputs if self.downsample is None else self.downsample(inputs)
        return self.relu(output + residual)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2,
                 dropout=0.2, use_weight_norm=True):
        super().__init__()
        layers = []
        for level, out_channels in enumerate(num_channels):
            dilation = 2 ** level
            in_channels = num_inputs if level == 0 else num_channels[level - 1]
            layers.append(TemporalBlock(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation,
                dropout=dropout,
                use_weight_norm=use_weight_norm,
            ))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)
