import numpy as np
import torch


def Clibloss(output):
    d_out = output['depth']
    lab_zeros = torch.zeros_like(d_out)

    return torch.nn.MSELoss()(d_out, lab_zeros)
