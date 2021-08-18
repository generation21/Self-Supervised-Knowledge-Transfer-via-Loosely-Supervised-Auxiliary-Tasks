import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import time
import sys
import torch.nn.functional as F
from utils import AverageMeter, calculate_accuracy

def CrossEntropy(predicted, target):
    return torch.mean(torch.sum(-nn.Softmax()(target) * torch.nn.LogSoftmax()(predicted), 1))

def model_fit(x_pred, x_output,loss_method, singleLabel):
    if loss_method == 'f':
        if not singleLabel:
            x_output = nn.Softmax()(x_output)
            x_pred = nn.Softmax()(x_pred)
            x_output_onehot = x_output
        if singleLabel:
            x_pred = nn.Softmax()(x_pred)
            x_output_onehot = torch.zeros((len(x_output), x_pred.size(1))).cuda()
            x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)
        
        loss = x_output_onehot * (1 - x_pred)**2 * torch.log(x_pred + 1e-20)
        loss = torch.mean(torch.sum(-loss, dim=1))
    elif loss_method == 'ce':
        if singleLabel:
            loss = F.cross_entropy(x_pred, x_output)
        if not singleLabel:
            loss = CrossEntropy(x_pred, x_output)

    return loss


def loss_fn_kd(Targetoutputs,auxiliaryOutput1, auxiliaryOutput2, imagenet_sourceOutput, places_sourceOutput, labels, opt):
    TargetsingleLabel = False if opt.dataset == 'voc' else True
    Target_loss = model_fit(Targetoutputs, labels, opt.classifier_loss_method, singleLabel=TargetsingleLabel)
    Auxiliary_imagenet = 0
    Auxiliary_places = 0
    if imagenet_sourceOutput is not None:
        Auxiliary_imagenet = model_fit(auxiliaryOutput1, imagenet_sourceOutput, opt.auxiliary_loss_method, singleLabel=False)
    if places_sourceOutput is not None:
        Auxiliary_places = model_fit(auxiliaryOutput2, places_sourceOutput, opt.auxiliary_loss_method, singleLabel=False)
    totalLoss = Target_loss + Auxiliary_imagenet + Auxiliary_places
    
    return totalLoss

def val_epoch(epoch, data_loader, model, sourceModels, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))
    source_spatial_transforms = transforms.Compose([
            transforms.Scale(224)
        ])
    model.eval()
    sourceImagenet, sourcePlaces = sourceModels
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            
            if not opt.no_cuda:
                targets = targets.cuda()
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs, auxiliary_image_output, auxiliary_places_output = model(inputs)
         
            if opt.isSource:
                center_images = inputs[:, :, int(inputs.size(2)/2), :, :]
                
                prob1 = None
                prob2 = None
                if opt.multi_source:
                    prob1 = sourceImagenet(center_images)
                    prob2 = sourcePlaces(center_images)
                elif (not opt.multi_source) and opt.sourceKind == 'imagenet':
                    prob1 = sourceImagenet(center_images)
                elif (not opt.multi_source) and opt.sourceKind == 'places':
                    prob2 = sourcePlaces(center_images)
                loss = loss_fn_kd(outputs,auxiliary_image_output, auxiliary_places_output, prob1, prob2, targets, opt)
            else:
                loss = loss_fn_kd(outputs, None, None, None, None, targets, opt)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch,
                    i + 1,
                    len(data_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    acc=accuracies))

        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return losses.avg
