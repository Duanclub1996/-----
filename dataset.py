import torchvision
from pathlib import Path
from torch.utils import data
import torchvision.transforms as transforms


def get_Dataloader(mode,batch_size):
    transform = transforms.Compose([
                                    transforms.Pad(4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32),
                                    transforms.ToTensor()])
    
    if Path('./data/').exists() == False :
        Path('./data/').mkdir()
    else:
        pass

    if mode == 'train':
        _dataset = torchvision.datasets.CIFAR10(root='../homework-9/data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)
    elif mode == 'val':
        _dataset =   torchvision.datasets.CIFAR10(root='../homework-9/data/',
                                            train=False, 
                                            transform=transforms.ToTensor(),
                                             download=True)
    elif mode == 'test':
        _dataset =  torchvision.datasets.CIFAR10(root='../homework-9/data/',
                                            train=False, 
                                            transform=transforms.ToTensor(),
                                             download=True)
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    data_loader = data.DataLoader(_dataset,batch_size = batch_size,shuffle=shuffle)
    
    return data_loader