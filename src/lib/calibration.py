from models.calib_loss import Clibloss


class Calibration():
    def __init__(self, optimizer=None):
        self.cmode = 0
        self.optimizer = None

    def step(self, frame, model):
        if self.cmode == 0:
            out = model(frame)[-1]
            loss = Clibloss(out)

            if loss.item() < 0.1:
                self.cmode = 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
