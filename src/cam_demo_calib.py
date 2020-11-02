from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import cv2
import numpy as np
import torch
import torch.utils.data
from opts import opts
from model import create_model
from calibration import Calibration
from utils.debugger import Dcam
from utils.image import get_affine_transform, transform_preds
from utils.eval import get_preds, get_preds_3d

image_ext = ['jpg', 'jpeg', 'png']
mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)


def is_image(file_name):
    ext = file_name[file_name.rfind('.') + 1:].lower()
    return ext in image_ext


def demo_image(image, model, opt):
    s = max(image.shape[0], image.shape[1]) * 1.0
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    trans_input = get_affine_transform(
        c, s, 0, [opt.input_w, opt.input_h])
    inp = cv2.warpAffine(image, trans_input, (opt.input_w, opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp / 255. - mean) / std
    inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    inp = torch.from_numpy(inp).to(opt.device)
    out = model(inp)[-1]
    preds, amb_idx = get_preds(out['hm'].detach().cpu().numpy())
    pred = preds[0]
    pred = transform_preds(pred, c, s, (opt.output_w, opt.output_h))
    pred_3d, ignore_idx = get_preds_3d(out['hm'].detach().cpu().numpy(),
                                       out['depth'].detach().cpu().numpy(),
                                       amb_idx)

    pred_3d = pred_3d[0]
    ignore_idx = ignore_idx[0]

    return image, pred, pred_3d, ignore_idx


def prog_img(frame, opt):
    s = max(frame.shape[0], frame.shape[1]) * 1.0
    c = np.array([frame.shape[1] / 2., frame.shape[0] / 2.], dtype=np.float32)
    trans_input = get_affine_transform(
        c, s, 0, [opt.input_w, opt.input_h])
    inp = cv2.warpAffine(frame, trans_input, (opt.input_w, opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp / 255. - mean) / std
    inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    inp = torch.from_numpy(inp).to(opt.device)

    return inp


def main(opt):
    camera = cv2.VideoCapture(0)
    opt.heads['depth'] = opt.num_output
    if opt.load_model == '':
        opt.load_model = '../models/fusion_3d_var.pth'
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
    else:
        opt.device = torch.device('cpu')

    if opt.demo != "":
        clb_img = cv2.imread(opt.demo)

    model, _, _ = create_model(opt)
    model = model.to(opt.device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    CLB = Calibration(optimizer=optimizer)

    debugger = Dcam()
    k = 0

    while debugger.loop_on:
        ret, frame = camera.read()
        if frame is None:
            return print("***No Camera Connecting***")

        if CLB.cmode == 0:
            if opt.demo != "":
                frame = clb_img
            inp = prog_img(frame, opt)
            CLB.step(inp, model)
            showimg = cv2.putText(frame, "Spread Your arms", (0, int(frame.shape[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 8)
            cv2.imshow('img', showimg)
        else:
            image, pred, pred_3d, ignore_idx = demo_image(frame, model, opt)

            debugger.add_img(image)
            debugger.add_point_2d(pred, (255, 0, 0))
            debugger.add_point_3d(pred_3d, 'b', ignore_idx=ignore_idx)
            debugger.realtime_show(k)
            debugger.destroy_loop()
            debugger.show_all_imgs()

        k = cv2.waitKey(10)
        if k == 27:
            debugger.loop_on = 0

        del ret, frame

    cv2.destroyAllWindows()
    camera.release()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
