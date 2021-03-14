import torch
import torchvision
from torchvision import transforms

import os
import pickle
from PIL import Image

dataset_folder_path = './dataset'
if not os.path.exists(dataset_folder_path):
    os.makedirs(dataset_folder_path)
    
    for split in [True, False]:
        torchvision.datasets.CIFAR100(
            root = dataset_folder_path,
            train=True,
            download=True)

import numpy as np


    
class CIFAR_100:
    """
    args:
    split - train or test
    path2DS - path to dataset
    
    """
    def __init__(self, path2DS, split = 'train', batch_size = 1):
        split = split.lower()
        assert split in ['train','test']
        
        self.path2DS    = path2DS
        self.batch_size = batch_size
        
        with open(os.path.join(self.path2DS,'cifar-100-python/{}'.format(split)), 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        
        self.labels   = data['fine_labels']
        self.data     = data['data'].reshape(-1,3,32,32).transpose(0,2,3,1)
        self.indexes  = np.arange(len(self.data))
        
        self.transformation = transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ])
    
    def reset_indexes(self):
        self.indexes  = np.random.permutation(self.indexes)

    def get_batch(self, i):
        indexes      = self.indexes[i*self.batch_size:(i+1)*self.batch_size]
        labels___    = [self.labels[index] for index in indexes]
        # ------------------------------------------------------------------
        batch_labels = torch.tensor(labels___         , dtype = torch.long)
        tensor_batch = []
        for index in indexes:
            PIL_image       = Image.fromarray(self.data[index])
            processed_image = self.transformation(PIL_image)
            tensor_batch.append(processed_image)
        
        batch_images = torch.stack(tensor_batch, 0)
        
        return batch_images, batch_labels
            
    def length(self):
        return len(self.indexes)