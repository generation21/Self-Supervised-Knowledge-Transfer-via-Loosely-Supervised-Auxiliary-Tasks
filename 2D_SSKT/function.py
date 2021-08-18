import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import AverageMeter, get_ap_score, accuracy

def CrossEntropy(predicted, target):
    return torch.mean(torch.sum(-nn.Softmax()(target) * torch.nn.LogSoftmax()(predicted), 1))
    
def model_fit(pred, target,loss_method, singleLabel, T=3):
 
    if loss_method == 'ce':
        if singleLabel:
            loss = F.cross_entropy(pred, target)
        if not singleLabel:
            loss =  T*T*CrossEntropy(pred/ T, target/ T)
    elif loss_method == 'kd':
        loss = T*T*nn.KLDivLoss()(F.log_softmax(pred / T, dim=1), F.softmax(target / T, dim=1))
        
    elif loss_method == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss = criterion(pred, target)
    return loss


def loss_fn(Targetoutputs,auxiliaryOutput1, auxiliaryOutput2, imagenet_sourceOutput, places_sourceOutput, labels, opt):
    TargetsingleLabel = False if opt.dataset == 'voc' else True
    Target_loss = model_fit(Targetoutputs, labels, opt.classifier_loss_method, singleLabel=TargetsingleLabel, T=opt.T)
    Auxiliary_imagenet = 0
    Auxiliary_places = 0
    if imagenet_sourceOutput is not None:
  
        Auxiliary_imagenet = model_fit(auxiliaryOutput1, imagenet_sourceOutput, opt.auxiliary_loss_method, singleLabel=False, T=opt.T)
    if places_sourceOutput is not None:
        Auxiliary_places = model_fit(auxiliaryOutput2, places_sourceOutput, opt.auxiliary_loss_method, singleLabel=False, T=opt.T)
    totalLoss = Target_loss + opt.alpha * (Auxiliary_imagenet + Auxiliary_places)
    
    return Target_loss, Auxiliary_imagenet, Auxiliary_places, totalLoss




def train(model, device, train_loader, optimizer, epoch, lamdaValue, opt):
    # set model as training mode
    targetNet, sourceImagenet, sourcePlaces = model
    targetNet.train()
    if opt.isSource:
        if opt.multi_source:
            sourceImagenet = sourceImagenet.train()
            sourcePlaces = sourcePlaces.train()
        elif (not opt.multi_source) and opt.sourceKind == 'imagenet':
            sourceImagenet = sourceImagenet.train()
        elif (not opt.multi_source) and opt.sourceKind == 'places365':
            sourcePlaces = sourcePlaces.train()

    losses = AverageMeter()
    Targetscores = AverageMeter()


    targetloss = AverageMeter()
    auxiliaryimagenet = AverageMeter()
    auxiliaryplaces = AverageMeter()

    softmax = nn.Softmax()
    sigmoid = torch.nn.Sigmoid()
    upsample = nn.Upsample(scale_factor=7/3, mode='bilinear')
    N_count = 0  
    for batch_idx, (images, y) in enumerate(train_loader):
        if opt.dataset == 'voc':
            y = y.float()
        if opt.dataset == 'places365':
            y = y.view(-1, )
        images, y = images.to(device), y.to(device)
        N_count+= images.size(0)
      
        optimizer.zero_grad()
        if opt.isSource:
            targetOutput, auxiliaryOutput1, auxiliaryOutput2 = targetNet(images)   
        else:
            targetOutput = targetNet(images)
     
        upsample_dataset = ['cifar10', 'cifar100', 'stl10']
        if opt.isSource:
            with torch.no_grad():
        
                if opt.dataset in upsample_dataset:
                    images = images.detach()              
                    images = upsample(images)
                
                prob1 = None
                prob2 = None
                if opt.multi_source:
                    prob1 = sourceImagenet(images)
                    prob2 = sourcePlaces(images)
                elif (not opt.multi_source) and opt.sourceKind == 'imagenet':
                    prob1 = sourceImagenet(images)
                elif (not opt.multi_source) and opt.sourceKind == 'places365':
                    prob2 = sourcePlaces(images)
          
            targetloss, auxiliary_imagenet, auxiliary_places, loss = loss_fn(targetOutput,auxiliaryOutput1, auxiliaryOutput2, prob1, prob2, y, opt)
        

                
        else:
            singleLabel = False if opt.dataset == 'voc' else True
            loss = model_fit(targetOutput, y, opt.classifier_loss_method, singleLabel=singleLabel, T=opt.T)

        losses.update(loss.item(), images.size()[0])

        y_pred = torch.max(targetOutput, 1)[1]  
        if opt.dataset == 'voc':
            step_score = get_ap_score(torch.Tensor.cpu(y).detach().numpy(), torch.Tensor.cpu(sigmoid(targetOutput)).detach().numpy())
        else:
            step_score = accuracy(targetOutput.data, y.data, topk=(1,))[0]
        Targetscores.update(step_score,images.size()[0])        
      
        loss.backward()
        optimizer.step()

        if (batch_idx) % 10 == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), losses.avg, Targetscores.avg))
            
   
    return losses, Targetscores


def validation(model, device, optimizer, test_loader, lamdaValue, opt):
    targetNet, sourceImagenet, sourcePlaces = model
    targetNet.eval()
    if opt.isSource:
        if opt.multi_source:
            sourceImagenet = sourceImagenet.eval()
            sourcePlaces = sourcePlaces.eval()
        elif (not opt.multi_source) and opt.sourceKind == 'imagenet':
            sourceImagenet = sourceImagenet.eval()
        elif (not opt.multi_source) and opt.sourceKind == 'places365':
            sourcePlaces = sourcePlaces.eval()

    accs = AverageMeter()
    losses = AverageMeter()

    softmax = nn.Softmax()
    sigmoid = torch.nn.Sigmoid()
    upsample = nn.Upsample(scale_factor=7/3, mode='bilinear')
    with torch.no_grad():
        for images, y in test_loader:
            # distribute data to device
            images, y = images.to(device), y.to(device)
            if opt.dataset == 'places365':
                y = y.view(-1, )
            if opt.dataset == 'voc':
                y = y.float()
            if opt.isSource:
                targetOutput, auxiliaryOutput1, auxiliaryOutput2 = targetNet(images)  
            else:
                targetOutput = targetNet(images)
         
            if opt.isSource:
                images = upsample(images)
                prob1 = None
                prob2 = None
                if opt.multi_source:
                    prob1 = sourceImagenet(images)
                    prob2 = sourcePlaces(images)
                elif (not opt.multi_source) and opt.sourceKind == 'imagenet':
                    prob1 = sourceImagenet(images)
                elif (not opt.multi_source) and opt.sourceKind == 'places365':
                    prob2 = sourcePlaces(images)
          
                targetloss, auxiliary_imagenet, auxiliary_places, loss = loss_fn(targetOutput,auxiliaryOutput1, auxiliaryOutput2, prob1, prob2, y, opt)

            else:
                singleLabel = False if opt.dataset == 'voc' else True
                loss = model_fit(targetOutput, y, opt.classifier_loss_method, singleLabel=singleLabel, T=opt.T)
            losses.update(loss.item(), images.size()[0])                
       
            if opt.dataset == 'voc':
                prec = get_ap_score(torch.Tensor.cpu(y).detach().numpy(), torch.Tensor.cpu(sigmoid(targetOutput)).detach().numpy())
            else:
                prec = accuracy(targetOutput.data, y.data, topk=(1,))[0]
            accs.update(prec.item(), images.size()[0])
   
        

    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(test_loader.dataset), losses.avg, accs.avg))
  
    
    return losses, accs

