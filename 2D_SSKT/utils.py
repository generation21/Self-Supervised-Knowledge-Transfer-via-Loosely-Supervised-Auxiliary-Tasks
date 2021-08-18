import os
from datetime import datetime
import numpy as np
import json
import csv
from sklearn.metrics import average_precision_score
import torch

def save_path(opt):
    savePath = opt.result
    date = datetime.today().strftime("%Y%m%d%H%M") 
    if opt.isSource:
        sourceKind = opt.sourceKind
        if opt.multi_source:
            sourceKind = 'I+P'
        date_method = opt.dataset + '/SSKT/' + opt.model + '_' + opt.classifier_loss_method \
        + '_' + sourceKind + '_' + opt.source_arch + '_' + opt.auxiliary_loss_method \
        + '_' + str(opt.transfer_module) + '_' + str(opt.T) + '_' + str(opt.alpha)  + '/' + date
    
    else:
        date_method = opt.dataset + '/scratch/' + opt.model + '_' + opt.classifier_loss_method  + '/' + date
    
    if not os.path.exists(os.path.join(savePath, date_method)):
        os.makedirs(os.path.join(savePath, date_method))
    with open(os.path.join(savePath, date_method, 'opt.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)
    print(os.path.join(savePath, date_method))
    save_model_path = os.path.join(savePath, date_method, opt.save_model_path)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    
    if not os.path.exists(os.path.join(savePath, date_method,'logfile')):
        os.makedirs(os.path.join(savePath, date_method,'logfile'))
    
    return savePath, date_method, save_model_path

class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

def get_ap_score(y_true, y_scores):
    """
    Get average precision score between 2 1-d numpy arrays
    
    Args:
        y_true: batch of true labels
        y_scores: batch of confidence scores
=
    Returns:
        sum of batch average precision
    """
    scores = 0.0
    
    for i in range(y_true.shape[0]):
        scores += average_precision_score(y_true = y_true[i], y_score = y_scores[i])
    
    return scores / y_true.shape[0]

def encode_labels(target):
    """
    Encode multiple labels using 1/0 encoding 
    
    Args:
        target: xml tree file
    Returns:
        torch tensor encoding labels as 1/0 vector
    """
    object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']
    ls = target['annotation']['object']
  
    j = []
    if type(ls) == dict:
        if int(ls['difficult']) == 0:
            j.append(object_categories.index(ls['name']))
  
    else:
        for i in range(len(ls)):
            if int(ls[i]['difficult']) == 0:
                j.append(object_categories.index(ls[i]['name']))
    
    k = np.zeros(len(object_categories))
    k[j] = 1
  
    return torch.from_numpy(k)
