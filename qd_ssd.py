#from data import *
from data_util.voc0712 import TSVDetection
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
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data_util import VOCDetection, BaseTransform
import torch.utils.data as data
import logging
from pprint import pformat

from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
import os.path as op

from tsv_io import TSVDataset, tsv_reader, tsv_writer
from tqdm import tqdm


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

def test_net(pred_file, net, cuda, dataset, labelmap, thresh=0.05):
    num_images = len(dataset)
    def gen_rows():
        for i in tqdm(range(num_images)):
            key, im, gt, h, w = dataset.pull_item(i)

            x = Variable(im.unsqueeze(0))
            if cuda:
                x = x.cuda()
            detections = net(x).data
            
            rects = []
            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.dim() == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
                for r in cls_dets:
                    if r[-1] < thresh:
                        continue
                    rect = {}
                    rect['rect'] = list(map(float, r[:4]))
                    rect['conf'] = float(r[-1])
                    rect['class'] = labelmap[j - 1]
                    rects.append(rect)
            from qd_common import json_dump
            yield key, json_dump(rects)
    tsv_writer(gen_rows(), pred_file)

def ssd_pipeline(**kwargs):
    kwargs['data'] = 'voc20'
    kwargs['net'] = 'vgg'
    kwargs['expid'] = 'test'
    kwargs['force_evaluate'] = True
    kwargs['ovthresh'] = [0.5]
    kwargs['cuda'] = True
    kwargs['gamma'] = 0.1
    kwargs['weight_decay'] = 5e-4
    kwargs['momentum'] = 0.9
    kwargs['save_folder'] = 'weights/'
    kwargs['lr'] = 1e-3
    kwargs['start_iter'] = 0
    kwargs['num_workers'] = 0
    kwargs['resume'] = True
    kwargs['max_iter'] = 1
    kwargs['confidence_threshold'] = 0.01
    kwargs['force_train'] = True
    #kwargs['confidence_threshold'] = 0.7
    kwargs['cuda'] = True

    t = SSDTrain(**kwargs)
    t.ensure_train()
    pred_file = t.ensure_predict()
    t.evaluate(pred_file)

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

from qd_pytorch import TorchTrain
class SSDTrain(TorchTrain):
    def __init__(self, **kwargs):
        super(SSDTrain, self).__init__(**kwargs)
    
    def train(self):
        kwargs = self.kwargs

        from process_tsv import TSVDataset
        tsv_dataset = TSVDataset(self.data)
        tsv_file = tsv_dataset.get_data('train')
        labelmap = tsv_dataset.get_labelmap_file()
        
        cfg = kwargs
        voc = {
            'num_classes': 21,
            'lr_steps': (80000, 100000, 120000),
            'max_iter': 120000,
            'feature_maps': [38, 19, 10, 5, 3, 1],
            'min_dim': 300,
            'steps': [8, 16, 32, 64, 100, 300],
            'min_sizes': [30, 60, 111, 162, 213, 264],
            'max_sizes': [60, 111, 162, 213, 264, 315],
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            'variance': [0.1, 0.2],
            'clip': True,
            'name': 'VOC',
        }
        for k in voc:
            if k in cfg:
                continue
            cfg[k] = voc[k]

        MEANS = (104, 117, 123)
        dataset = TSVDetection(tsv_file=tsv_file,
                labelmap=labelmap,
                transform=SSDAugmentation(cfg['min_dim'], MEANS))

        ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
        net = ssd_net

        if cfg['cuda']:
            net = torch.nn.DataParallel(ssd_net)
            cudnn.benchmark = True

        vgg_weights = torch.load(cfg['save_folder'] + 'vgg16_reducedfc.pth')
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
        cfg['batch_size'] = 32

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
                logging.info('Saving state, iter: {}'.format(iteration))
                model_file = op.join(self.output_folder, 'snapshot',
                        'model_iter_{}.pth.tar'.format(iteration + 1))
                torch.save(ssd_net.state_dict(), model_file)
        model_file = op.join(self.output_folder, 'snapshot', 
                'model_iter_{}.pth.tar'.format(iteration + 1))
        from qd_pytorch import torch_save
        torch_save(ssd_net.state_dict(), model_file)

    def predict(self, model_file, predict_result_file):
        train_dataset = TSVDataset(self.data)
        labelmap = train_dataset.load_labelmap()

        num_classes = len(labelmap) + 1                      # +1 for background
        net = build_ssd('test', 300, num_classes)            # initialize SSD
        net.load_state_dict(torch.load(model_file))
        net.eval()

        test_dataset = TSVDataset(self.test_data)
        dataset_mean = (104, 117, 123)
        dataset = TSVDetection(test_dataset.get_data('test'),
                train_dataset.get_labelmap_file(),
                BaseTransform(300, dataset_mean))
        if self.kwargs['cuda']:
            net = net.cuda()
            cudnn.benchmark = True
        pred_file = self.kwargs['pred_file']
        test_net(pred_file, net, self.kwargs['cuda'], dataset,
                labelmap, thresh=self.kwargs['confidence_threshold'])
        return pred_file

if __name__ == '__main__':
    from qd_common import init_logging, parse_general_args
    init_logging()

    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)

