import numpy as np
import torch


def Clibloss(output):
    d_size = output.size()
    lab_zeros = torch.zeros_like(d_size)

    return torch.nn.MSELoss()(output, lab_zeros)