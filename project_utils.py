import torch
from torch.utils.data import DataLoader 
from torchvision import transforms, datasets
import torchvision.models as models

from enum import Enum

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'Training': transforms.Compose ([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))    
        ]),
        'ValidationAndTesting': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))    
        ])
    }

    image_datasets = {
        'Training': datasets.ImageFolder (train_dir, transform = data_transforms['Training']),
        'Validation': datasets.ImageFolder (valid_dir, transform = data_transforms['ValidationAndTesting']),
        'Testing': datasets.ImageFolder (test_dir, transform = data_transforms['ValidationAndTesting'])
    }

    dataloaders = {
        'Training': DataLoader(image_datasets['Training'], batch_size=32, shuffle=True),
        'Validation': DataLoader(image_datasets['Validation'], batch_size=32, shuffle=True),
        'Testing': DataLoader(image_datasets['Testing'], batch_size=32, shuffle=True)
    }

    return image_datasets, dataloaders

class ArchName(Enum):
    vgg19 = 'vgg19'
    alexnet = 'alexnet'

    def load_model(self):
        if self == ArchName.vgg19:
            return models.vgg19(pretrained=True)
        else:
            return models.alexnet(pretrained=True)

    def get_input_layer_size(self):
        if self == ArchName.vgg19:
            return 25088
        else:
            return 9216

    def __str__(self):
        return self.value

def create_device(is_gpu):
    return torch.device('cuda' if is_gpu and torch.cuda.is_available() else 'cpu')
