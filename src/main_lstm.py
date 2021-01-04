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

dataset_factory = {
    'mpii': MPII,
    'coco': COCO,
    'fusion_3d': Fusion3D,
    'lstm': H36M,
    'conv3d': H36M
}

task_factory = {
    'human2d': (train, val),
    'human3d': (train_3d, val_3d),
    "lstm": (train_lstm, val_lstm),
    "conv3d": (train_conv3d, val_conv3d)
}


def main(opt):
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print('Cudnn is disabled.')

    timestep = 4

    logger = Logger(opt)
    opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))

    Dataset = dataset_factory[opt.dataset]
    LstmData = SeqH36m(Dataset(opt, 'train', 1), timestep)
    train, val = task_factory[opt.task]

    if opt.task == "conv3d":
        model, optimizer, start_epoch = create_conv3d(opt, timestep)
    else:
        model, optimizer, start_epoch = create_lstm(opt, timestep)

    if len(opt.gpus) > 1:
        model = torch.nn.DataParallel(
            model, device_ids=opt.gpus).cuda(
            opt.device)
    else:
        model = model.cuda(opt.device)

    val_loader = torch.utils.data.DataLoader(
        SeqH36m(Dataset(opt, 'val', 1), timestep),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    if opt.test:
        log_dict_train, preds = val(0, opt, val_loader, model)
        sio.savemat(os.path.join(opt.save_dir, 'preds.mat'),
                    mdict={'preds': preds})
        return

    train_loader = torch.utils.data.DataLoader(
        LstmData,
        batch_size=opt.batch_size * len(opt.gpus),
        shuffle=True,  # if opt.debug == 0 else False,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    best = -1
    for epoch in range(start_epoch, opt.num_epochs + 1):
        mark = epoch if opt.save_all_models else 'last'
        log_dict_train, _ = train(epoch, opt, train_loader, model, optimizer, timestep)
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_lstm_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            log_dict_val, preds = val(epoch, opt, val_loader, model, timestep)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] > best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_lstm_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_lstm_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    # logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
