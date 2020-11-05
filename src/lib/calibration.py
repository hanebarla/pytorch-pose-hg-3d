import torch
import numpy as np
import cv2
from models.calib_loss import ImgLoss
from models.calib_loss import Clibloss


class Calibration():
    def __init__(self, optimizer=None, opt=None):
        self.cmode = 0
        self.optimizer = optimizer
        self.opt = opt

    def step(self, frame, model):
        if self.cmode == 0:
            out = model(frame)[-1]
            if self.opt.demo == "":
                loss = Clibloss(out)
            else:
                loss = ImgLoss(out)

            if loss.item() < 1e-4:
                self.cmode = 1
                model.eval()
                cv2.destroyWindow('img')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
