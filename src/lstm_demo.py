from __future__ import absolute_import, barry_as_FLUFL
from __future__ import division
from __future__ import print_function

import _init_paths

import cv2
import numpy as np
import torch
import torch.utils.data
from opts import opts
from model import create_model, create_lstm
from utils.debugger import Debugger, Dcam
from utils.image import get_affine_transform, transform_preds
from utils.eval import get_preds, get_preds_3d

image_ext = ['jpg', 'jpeg', 'png']
mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)


def is_image(file_name):
    ext = file_name[file_name.rfind('.') + 1:].lower()
    return ext in image_ext


def demo_image(image, model, opt, timestep):
    inps = []
    s = None
    c = None
    hidden = None
    for t in range(timestep):
        s = max(image[t].shape[0], image[t].shape[1]) * 1.0
        c = np.array([image[t].shape[1] / 2., image[t].shape[0] / 2.], dtype=np.float32)
        trans_input = get_affine_transform(
            c, s, 0, [opt.input_w, opt.input_h])
        inp = cv2.warpAffine(image[t], trans_input, (opt.input_w, opt.input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp / 255. - mean) / std
        inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        inp = torch.from_numpy(inp).to(opt.device)
        inps.append(inp)
    out = model(inps, hidden)[-1][-1]
    preds, amb_idx = get_preds(out['hm'].detach().cpu().numpy())
    pred = preds[0]
    pred = transform_preds(pred, c, s, (opt.output_w, opt.output_h))
    pred_3d, ignore_idx = get_preds_3d(out['hm'].detach().cpu().numpy(),
                                       out['depth'].detach().cpu().numpy(),
                                       amb_idx)

    pred_3d = pred_3d[0]
    ignore_idx = ignore_idx[0]

    return image, pred, pred_3d, ignore_idx


def main(opt):
    if opt.mode == "video":
        assert opt.video != '', "No demo path"
        camera = cv2.VideoCapture(opt.video)
    else:
        camera = cv2.VideoCapture(0)
    opt.heads['depth'] = opt.num_output
    if opt.load_model == '':
        opt.load_model = '../models/fusion_3d_var.pth'
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
    else:
        opt.device = torch.device('cpu')

    timestep = 4

    model, _, _ = create_lstm(opt, timestep)
    model = model.to(opt.device)
    model.eval()

    debugger = Dcam()
    k = 0
    input_imgs = []

    while debugger.loop_on:
        ret, frame = camera.read()
        if frame is None:
            return print("***No Camera Connecting***")

        if len(input_imgs) < timestep:
            input_imgs.append(frame)
        elif len(input_imgs) == timestep:
            image, pred, pred_3d, ignore_idx = demo_image(input_imgs, model, opt, timestep)

            debugger.add_img(image)
            debugger.add_point_2d(pred, (255, 0, 0))
            debugger.add_point_3d(pred_3d, 'b', ignore_idx=ignore_idx)
            debugger.realtime_show(k)
            debugger.destroy_loop()
            debugger.show_all_imgs()

            k = cv2.waitKey(10)
            if k == 27:
                debugger.loop_on = 0
        else:
            raise ValueError

    cv2.destroyAllWindows()
    camera.release()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
