import torch
import numpy as np
from torch._C import dtype
from utils.image import flip, shuffle_lr
from utils.eval import accuracy, get_preds, mpjpe, get_preds_3d
import cv2
from progress.bar import Bar
from utils.debugger import Debugger
from models.losses import RegLoss, FusionLoss
import time


def step(split, epoch, opt, data_loader, model, optimizer=None, timestep=4):
    if split == 'train':
        model.train()
    else:
        model.eval()

    crit = torch.nn.MSELoss()
    crit_3d = FusionLoss(opt.device, opt.weight_3d, opt.weight_var)

    acc_idxs = data_loader.dataset.dataset.acc_idxs
    edges = data_loader.dataset.dataset.edges
    edges_3d = data_loader.dataset.dataset.edges_3d
    shuffle_ref = data_loader.dataset.dataset.shuffle_ref
    mean = data_loader.dataset.dataset.mean
    std = data_loader.dataset.dataset.std
    convert_eval_format = data_loader.dataset.dataset.convert_eval_format

    Loss, Loss3D = AverageMeter(), AverageMeter()
    Acc, MPJPE = AverageMeter(), AverageMeter()
    data_time, batch_time = AverageMeter(), AverageMeter()
    preds = []
    time_str = ''

    nIters = len(data_loader)
    bar = Bar('{}'.format(opt.exp_id), max=nIters)

    end = time.time()
    for i, batches in enumerate(data_loader):
        data_time.update(time.time() - end)
        loss = 0.0
        loss_3d_times = []

        for k in batches:
            if k != 'meta':
                for t in range(timestep):
                    batches[k][t] = batches[k][t].cuda(device=opt.device, non_blocking=True)

        out_rets = model(batches['input'])

        for t in range(timestep):
            gt_2d = batches['meta'][t]['pts_crop'].cuda(
                device=opt.device, non_blocking=True).float() / opt.output_h

            loss = crit(out_rets[t][-1]['hm'], batches['target'][t])
            loss_3d = crit_3d(
                out_rets[t][-1]['depth'], batches['reg_mask'][t], batches['reg_ind'][t],
                batches['reg_target'][t], gt_2d)
            loss_3d_times.append(loss_3d)
            for k in range(opt.num_stacks - 1):
                loss += crit(out_rets[t][k], batches['target'][t])
                loss_3d = crit_3d(
                    out_rets[t][-1]['depth'], batches['reg_mask'][t], batches['reg_ind'][t],
                    batches['reg_target'][t], gt_2d)
                loss_3d_times.append(loss_3d)
            loss += loss_3d

        if split == 'train':
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        else:
            for t in range(timestep):
                input_ = batches['input'][t].cpu().numpy().copy()
                input_[0] = flip(input_[0]).copy()[np.newaxis, ...]
                input_flip_var = torch.from_numpy(input_).cuda(
                    device=opt.device, non_blocking=True)
                output_flip_ = model(input_flip_var)
                output_flip = shuffle_lr(
                    flip(output_flip_[-1]['hm'].detach().cpu().numpy()[0]), shuffle_ref)
                output_flip = output_flip.reshape(
                    1, opt.num_output, opt.output_h, opt.output_w)
                output_depth_flip = shuffle_lr(
                    flip(output_flip_[-1]['depth'].detach().cpu().numpy()[0]), shuffle_ref)
                output_depth_flip = output_depth_flip.reshape(
                    1, opt.num_output, opt.output_h, opt.output_w)
                output_flip = torch.from_numpy(output_flip).cuda(
                    device=opt.device, non_blocking=True)
                output_depth_flip = torch.from_numpy(output_depth_flip).cuda(
                    device=opt.device, non_blocking=True)
                out_rets[t][-1]['hm'] = (out_rets[t][-1]['hm'] + output_flip) / 2
                out_rets[t][-1]['depth'] = (out_rets[t][-1]['depth'] + output_depth_flip) / 2
                # pred, amb_idx = get_preds(output[-1]['hm'].detach().cpu().numpy())
                # preds.append(convert_eval_format(pred, conf, meta)[0])

        for t in range(timestep):
            loss_3d = loss_3d_times[t]
            Loss.update(loss.item(), batches['input'][t].size(0))
            Loss3D.update(loss_3d.item(), batches['input'][t].size(0))
            Acc.update(accuracy(out_rets[t][-1]['hm'].detach().cpu().numpy(),
                                batches['target'][t].detach().cpu().numpy(), acc_idxs))
            mpeje_batch, mpjpe_cnt = mpjpe(out_rets[t][-1]['hm'].detach().cpu().numpy(),
                                           out_rets[t][-1]['depth'].detach().cpu().numpy(),
                                           batches['meta'][t]['gt_3d'].detach().numpy(),
                                           convert_func=convert_eval_format)
            MPJPE.update(mpeje_batch, mpjpe_cnt)

        batch_time.update(time.time() - end)
        end = time.time()

        if not opt.hide_data_time:
            time_str = ' |Data {dt.avg:.3f}s({dt.val:.3f}s)' \
                       ' |Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)

        Bar.suffix = '{split}: [{0}][{1}/{2}] |Total {total:} |ETA {eta:} '\
                     '|Loss {loss.avg:.5f} |Loss3D {loss_3d.avg:.5f}'\
                     '|Acc {Acc.avg:.4f} |MPJPE {MPJPE.avg:.2f}'\
                     '{time_str}'.format(epoch, i, nIters, total=bar.elapsed_td,
                                         eta=bar.eta_td, loss=Loss, Acc=Acc,
                                         split=split, time_str=time_str,
                                         MPJPE=MPJPE, loss_3d=Loss3D)
        if opt.print_iter > 0:
            if i % opt.print_iter == 0:
                print('{}| {}'.format(opt.exp_id, Bar.suffix))
        else:
            bar.next()
        if opt.debug >= 2:
            for t in range(timestep):
                output = out_rets[t]
                batch = batches[t]
                gt, amb_idx = get_preds(batches['target'].cpu().numpy())
                gt *= 4
                pred, amb_idx = get_preds(output[-1]['hm'].detach().cpu().numpy())
                pred *= 4
                debugger = Debugger(ipynb=opt.print_iter > 0, edges=edges)
                img = (
                    batch['input'][0].cpu().numpy().transpose(
                        1, 2, 0) * std + mean) * 256
                img = img.astype(np.uint8).copy()
                debugger.add_img(img)
                debugger.add_mask(
                    cv2.resize(batch['target'][0].cpu().numpy().max(axis=0),
                               (opt.input_w, opt.input_h)), img, 'target')
                debugger.add_mask(
                    cv2.resize(output[-1]['hm'][0].detach().cpu().numpy().max(axis=0),
                               (opt.input_w, opt.input_h)), img, 'pred')
                debugger.add_point_2d(gt[0], (0, 0, 255))
                debugger.add_point_2d(pred[0], (255, 0, 0))
                debugger.add_point_3d(
                    batch['meta']['gt_3d'].detach().numpy()[0],
                    'r',
                    edges=edges_3d)
                pred_3d, ignore_idx = get_preds_3d(output[-1]['hm'].detach().cpu().numpy(),
                                                   output[-1]['depth'].detach().cpu().numpy(),
                                                   amb_idx)
                debugger.add_point_3d(
                    convert_eval_format(
                        pred_3d[0]), 'b', edges=edges_3d)
                debugger.show_all_imgs(pause=False)
                debugger.show_3d()

    bar.finish()
    return {'loss': Loss.avg,
            'acc': Acc.avg,
            'mpjpe': MPJPE.avg,
            'time': bar.elapsed_td.total_seconds() / 60.}, preds


def train_conv3d(epoch, opt, train_loader, model, optimizer, timestep):
    return step('train', epoch, opt, train_loader, model, optimizer, timestep)


def val_conv3d(epoch, opt, val_loader, model, timestep):
    return step('val', epoch, opt, val_loader, model, timestep)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
