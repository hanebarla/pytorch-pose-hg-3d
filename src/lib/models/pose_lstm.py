import torch.nn as nn


class PoseLSTM(nn.Module):
    def __init__(self, i_dim=10, h_dim=10, l_num=10) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=i_dim, hidden_size=h_dim, num_layers=l_num)

    def forward(self, x, hp, cp):
        y, (hn, cn) = self.lstm(x, (hp, cp))

        return y, (hn, cn)


def get_pose_lstm(opt, idim, hdim, lnum):
    return PoseLSTM(idim, hdim, lnum)