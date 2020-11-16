import torch
import torch.nn as nn


class PoseLSTM(nn.Module):
    def __init__(self, i_dim=10, h_dim=10, l_num=10) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=i_dim, hidden_size=h_dim, num_layers=l_num)

    def forward(self, x):
        _, lstm_out = self.lstm(x)
