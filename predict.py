"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOCDetection, BaseTransform
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

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from tsv_io import TSVDataset, tsv_reader, tsv_writer
from tqdm import tqdm

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

def predict(**kwargs):
    train_data = 'voc20'
    train_dataset = TSVDataset(train_data)
    labelmap = train_dataset.load_labelmap()

    # load net
    num_classes = len(labelmap) + 1                      # +1 for background
    net = build_ssd('test', 300, num_classes)            # initialize SSD
    net.load_state_dict(torch.load(kwargs['trained_model']))
    net.eval()

    test_data = 'voc20'
    test_dataset = TSVDataset(test_data)
    from data.voc0712 import TSVDetection
    dataset_mean = (104, 117, 123)
    dataset = TSVDetection(test_dataset.get_data('test'),
            train_dataset.get_labelmap_file(),
            BaseTransform(300, dataset_mean))
    if kwargs['cuda']:
        net = net.cuda()
        cudnn.benchmark = True
    pred_file = kwargs['pred_file']
    # evaluation
    test_net(pred_file, net, kwargs['cuda'], dataset,
            labelmap, thresh=kwargs['confidence_threshold'])
    gt_file = test_dataset.get_data('test', 'label')
    from deteval import deteval_iter
    deteval_iter(tsv_reader(gt_file), pred_file,
            report_file=op.splitext(pred_file)[0] + '.report')

if __name__ == '__main__':
    from qd_common import init_logging, parse_general_args
    init_logging()

    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)

