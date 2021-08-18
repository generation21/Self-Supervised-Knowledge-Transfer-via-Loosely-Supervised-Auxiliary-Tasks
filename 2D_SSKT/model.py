import os


import torch
import torch.nn as nn

import torchvision.models as models


class SourceImageNet(nn.Module):
    def __init__(self,
                source_arch = 'resnet101'):
        super(SourceImageNet, self).__init__()
  
  
        readModel = models.__dict__[source_arch](pretrained=True)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']:
            self.add_module(module_name, getattr(readModel, module_name))

      
    def forward(self, x):
        with torch.no_grad():
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x

class SourcePlaces(nn.Module):
    def __init__(self, 
                source_arch = 'resnet101'):
        super(SourcePlaces, self).__init__()

        model_file = '/../Pretrained_model/places/%s_places365.pth.tar' % source_arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        model = models.__dict__[source_arch](num_classes=365)
        
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
 
        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']:
            self.add_module(module_name, getattr(model, module_name))

    def forward(self, x):
        with torch.no_grad():
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)        
       
        return x