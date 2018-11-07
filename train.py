from data import *
from data.voc0712 import TSVDetection
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import logging


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')
    train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                        type=str, help='VOC or COCO')
    parser.add_argument('--dataset_root', default=VOC_ROOT,
                        help='Dataset root directory path')
    parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                        help='Pretrained base model')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='Resume training at this iter')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='weights/',
                        help='Directory for saving checkpoint models')
    args = parser.parse_args()


    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                  "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    return args

def train(**kwargs):
    train_data = 'voc20'
    train_split = 'train'
    from process_tsv import TSVDataset
    tsv_dataset = TSVDataset(train_data)
    tsv_file = tsv_dataset.get_data(train_split)
    labelmap = tsv_dataset.get_labelmap_file()
    
    cfg = kwargs
    if cfg['dataset'] == 'COCO':
        if cfg['dataset_root'] == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            cfg['dataset_root'] = COCO_ROOT
        for k in coco:
            assert k not in cfg
            cfg[k] = coco[k]
        dataset = COCODetection(root=cfg['dataset_root'],
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    #elif cfg['dataset'] == 'VOC':
        #if cfg['dataset_root'] == COCO_ROOT:
            #parser.error('Must specify dataset if specifying dataset_root')
        #for k in voc:
            #assert k not in cfg
            #cfg[k] = voc[k]
        #dataset = VOCDetection(root=cfg['dataset_root'],
                               #transform=SSDAugmentation(cfg['min_dim'],
                                                         #MEANS))
    elif cfg['dataset']:
        for k in voc:
            assert k not in cfg
            cfg[k] = voc[k]
        dataset = TSVDetection(tsv_file=tsv_file,
                labelmap=labelmap,
                transform=SSDAugmentation(cfg['min_dim'], MEANS))


    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if cfg['cuda']:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if cfg['resume']:
        print('Resuming training, loading {}...'.format(cfg['resume']))
        ssd_net.load_weights(cfg['resume'])
        assert False, 'optimizer parameter is not loaded. Need a fix'
    else:
        vgg_weights = torch.load(cfg['save_folder'] + cfg['basenet'])
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if cfg['cuda']:
        net = net.cuda()

    if cfg['resume']:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=cfg['lr'],
            momentum=cfg['momentum'],
                          weight_decay=cfg['weight_decay'])
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, cfg['cuda'])

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    logging.info('Loading the dataset...')

    epoch_size = len(dataset) // cfg['batch_size']

    step_index = 0

    data_loader = data.DataLoader(dataset, cfg['batch_size'],
                                  num_workers=cfg['num_workers'],
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(cfg['start_iter'], cfg['max_iter']):
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, cfg['lr'], cfg['gamma'], step_index)

        t0 = time.time()
        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if cfg['cuda']:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        data_loading_time = time.time() - t0
        t0 = time.time()
        # forward
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        if iteration % 10 == 0:
            logging.info('data loading time {}'.format(data_loading_time))
            logging.info('timer: %.4f sec.' % (time.time() - t0))
            logging.info('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]))

        if iteration != 0 and iteration % 500 == 0:
            logging.info('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               cfg['save_folder'] + '' + cfg['dataset'] + '.pth')


def adjust_learning_rate(optimizer, base_lr, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = base_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def init_logging():
    logging.basicConfig(level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(funcName)10s(): %(message)s',
    datefmt='%m-%d %H:%M:%S',
    )

if __name__ == '__main__':
    init_logging()
    args = parse_args()
    kwargs = vars(args)
    train(**kwargs)
