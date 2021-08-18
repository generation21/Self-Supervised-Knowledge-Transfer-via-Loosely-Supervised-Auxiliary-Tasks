import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

def read_model(opt):
        # load the pre-trained weights
        if opt.sourceKind == "places":
            arch = opt.source_arch
            model_file = '../../../Pretrained_model/places/%s_places365.pth.tar' % arch
            if not os.access(model_file, os.W_OK):
                weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
                os.system('wget ' + weight_url)

            model = models.__dict__[arch](num_classes=365)
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            if opt.source_arch == "densenet161":
                state_dict = {str.replace(k,'auxiliary.','norm'): v for k,v in state_dict.items()}
                state_dict = {str.replace(k,'conv.','conv'): v for k,v in state_dict.items()}
                state_dict = {str.replace(k,'normweight','norm.weight'): v for k,v in state_dict.items()}
                state_dict = {str.replace(k,'normrunning','norm.running'): v for k,v in state_dict.items()}
                state_dict = {str.replace(k,'normbias','norm.bias'): v for k,v in state_dict.items()}
                state_dict = {str.replace(k,'convweight','conv.weight'): v for k,v in state_dict.items()}
            model.load_state_dict(state_dict)
       
        elif opt.sourceKind == "imagenet":
            model = models.__dict__[opt.source_arch](pretrained=True)
        return model
    
class SourceCNN(nn.Module):
    def __init__(self, opt):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(SourceCNN, self).__init__()
  
        readModel = read_model(opt)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']:
            self.add_module(module_name, getattr(readModel, module_name))

        in_features = readModel.fc.in_features
      
    def forward(self, x):
        cnn_embed_seq = []
        Feature = []
        # ResNet CNN
        with torch.no_grad():
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            b1 = self.layer1(x)
            b2 = self.layer2(b1)
            b3 = self.layer3(b2)
            b4 = self.layer4(b3)
            
            pool = self.avgpool(b4)
            x = pool.view(pool.size(0), -1)
            prob365 = self.fc(x)        
       
        return prob365

class SourceImageNet(nn.Module):
    def __init__(self, opt):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(SourceImageNet, self).__init__()

  
        readModel = models.__dict__[opt.source_arch](pretrained=True)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']:
            self.add_module(module_name, getattr(readModel, module_name))

      
    def forward(self, x):
        cnn_embed_seq = []
        Feature = []
        # ResNet CNN
        with torch.no_grad():
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            b1 = self.layer1(x)
            b2 = self.layer2(b1)
            b3 = self.layer3(b2)
            b4 = self.layer4(b3)
            
            pool = self.avgpool(b4)
            x = pool.view(pool.size(0), -1)
            prob365 = self.fc(x)        
       
        return prob365

class SourcePlaces(nn.Module):
    def __init__(self, opt):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(SourcePlaces, self).__init__()


        model_file = '../../../Pretrained_model/places/%s_places365.pth.tar' % opt.source_arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        model = models.__dict__[opt.source_arch](num_classes=365)
        
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        if opt.source_arch == "densenet161":
            state_dict = {str.replace(k,'auxiliary.','norm'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'conv.','conv'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'normweight','norm.weight'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'normrunning','norm.running'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'normbias','norm.bias'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'convweight','conv.weight'): v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)
 
        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']:
            self.add_module(module_name, getattr(model, module_name))

    def forward(self, x):
        cnn_embed_seq = []
        Feature = []
        # ResNet CNN
        with torch.no_grad():
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            b1 = self.layer1(x)
            b2 = self.layer2(b1)
            b3 = self.layer3(b2)
            b4 = self.layer4(b3)
            
            pool = self.avgpool(b4)
            x = pool.view(pool.size(0), -1)
            prob365 = self.fc(x)        
       
        return prob365
