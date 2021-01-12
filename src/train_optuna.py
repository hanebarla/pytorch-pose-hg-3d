from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from model import create_model, save_model, create_lstm, create_conv3d
from datasets.mpii import MPII
from datasets.coco import COCO
from datasets.fusion_3d import Fusion3D
from datasets.h36m import H36M, SeqH36m
from logger import Logger
from train import train, val
from train_3d import train_3d, val_3d
from train_lstm import train_lstm, val_lstm
from train_conv3d import train_conv3d, val_conv3d
import scipy.io as sio

import optuna
from main_lstm import main


TRIAL_SIZE = 100


def objective(trial):
    opt = opts().parse()
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-2)
    opt.lr = lr
    opt.weight_decay = weight_decay
    loss = main(opt)

    return loss


def optuning_hyper(obj):
    study = optuna.create_study()
    study.optimize(obj, n_trials=TRIAL_SIZE)

    return study


if __name__ == "__main__":
    study = optuning_hyper(objective)
    print(study.best_params)
