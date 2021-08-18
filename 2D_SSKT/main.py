# Task Transfer Learning
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import os
import matplotlib.pyplot as plt
import numpy as np

from opts import parse_opts
from model import SourceImageNet, SourcePlaces
from function import train, validation
from utils import save_path
from dataload import dataLoadFunc

from targetModel import generate_model
if __name__ == '__main__':
    opt = parse_opts()
    # torch.manual_seed(1)
    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    train_loader, valid_loader = dataLoadFunc(opt)

    targetCNN = generate_model(opt)
    targetCNN = targetCNN.to(device)
    sourceImagenet = None
    sourcePlaces = None
    if opt.isSource:
        if opt.multi_source:
            sourceImagenet = SourceImageNet(source_arch = opt.source_arch).to(device)
            sourcePlaces = SourcePlaces(source_arch = opt.source_arch).to(device)
        elif (not opt.multi_source) and opt.sourceKind == 'imagenet':
            sourceImagenet = SourceImageNet(source_arch = opt.source_arch).to(device)
        elif (not opt.multi_source) and opt.sourceKind == 'places':
            sourcePlaces = SourcePlaces(source_arch = opt.source_arch).to(device)

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        targetCNN = nn.DataParallel(targetCNN)
        parms = list(targetCNN.module.parameters())
    
        if opt.isSource:
            if opt.multi_source:
                sourceImagenet = nn.DataParallel(sourceImagenet)
                sourcePlaces = nn.DataParallel(sourcePlaces)
            elif (not opt.multi_source) and opt.sourceKind == 'imagenet':
                sourceImagenet = nn.DataParallel(sourceImagenet)
            elif (not opt.multi_source) and opt.sourceKind == 'places':
                sourcePlaces = nn.DataParallel(sourcePlaces)
  
    elif torch.cuda.device_count() == 1:
        print("Using", torch.cuda.device_count(), "GPU!")
        # Combine all EncoderCNN + DecoderRNN parameters
        parms = list(targetCNN.parameters())

    optimizer = torch.optim.SGD(parms, opt.lr,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)


    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []


    lamdaValue = np.ones((opt.epochs))

    savePath, date_method, save_model_path  = save_path(opt)
    writer = SummaryWriter(os.path.join(savePath, date_method,'logfile'))
    # start training
    for epoch in range(1, opt.epochs + 1):
        # train, test model
        train_losses, train_Target_scores, = train([targetCNN, sourceImagenet, sourcePlaces], device, train_loader, optimizer, epoch, lamdaValue[epoch - 1], opt)
        
        test_total_loss, test_total_score = validation([targetCNN, sourceImagenet, sourcePlaces], device, optimizer, valid_loader, lamdaValue[epoch - 1], opt)
        scheduler.step()

        # save results
        epoch_train_losses.append(np.mean(train_losses))
        epoch_train_scores.append(np.mean(train_Target_scores))
        epoch_test_losses.append(np.mean(test_total_loss))
        epoch_test_scores.append(np.mean(test_total_score))
                
        # plot average of each epoch loss value
        writer.add_scalar('Loss/train', np.mean(train_losses), epoch)
        writer.add_scalar('Loss/test', np.mean(test_total_loss), epoch)
        
        writer.add_scalar('scores/train', np.mean(train_Target_scores), epoch)
        writer.add_scalar('scores/test', test_total_score, epoch)
               
        if epoch % 5 == 0:
            torch.save({'state_dict': targetCNN.state_dict()}, os.path.join(save_model_path, 'targetCNN_lastest_epoch{}.pth'.format(epoch)))  # save spatial_encoder
            torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_lastest{}.pth'.format(epoch)))      # save optimizer
            print("Epoch {} model saved!".format(epoch))
            # save all train test results

    epoch_test_losses = np.array(epoch_test_losses)
    epoch_test_scores = np.array(epoch_test_scores)
    
        
    scoreTxtSave = open(os.path.join(savePath, date_method, 'TotalScore'), 'w')
    for i in range(0, len(epoch_test_scores)):
        data = 'epoch' + str(i + 1) + ' loss : ' + str(epoch_test_losses[i])+ ' score : ' + str(epoch_test_scores[i] * 100) + '\n'
        scoreTxtSave.write(data)
    scoreTxtSave.close()
    