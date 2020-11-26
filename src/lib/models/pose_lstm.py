import torch.nn as nn


class PoseLSTM(nn.Module):
    def __init__(self, i_dim=10, outdim=5) -> None:
        super().__init__()
        self.lstmin_linear = nn.Linear(i_dim, 10)
        self.lstm = nn.LSTM(input_size=10, hidden_size=10, batch_first=True)
        self.lstmout_linear = nn.Linear(10, i_dim)

    def forward(self, x, hidden):
        x = self.lstmin_linear(x)
        y, hd = self.lstm(x, hidden)
        y = self.lstmout_linear(y)

        return y, hd


def get_pose_lstm(opt, idim, hdim, lnum):
    return PoseLSTM(idim, hdim)