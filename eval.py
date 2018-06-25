from torchvision import datasets, models, transforms
import torch.optim as optim
import os
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from InceptionV3 import *


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.765, 0.547, 0.572], [0.142, 0.153, 0.170])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.765, 0.547, 0.572], [0.142, 0.153, 0.170])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.765, 0.547, 0.572], [0.142, 0.153, 0.170])])
}


device_ind = 1
batch_size = 12
data_dir = '/projects/melanoma/ISIC/Meredith/Data'
model_dir = '/projects/melanoma/ISIC/Meredith/saved_model/Xception.pt'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                                        for x in ['train','test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=
                                              {'train':True,'val':True,'test':False}[x], num_workers=8)
                                              for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:{}".format(device_ind)
                      if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

model_ft = torch.load(model_dir)
model_ft = model_ft.to(device)
model_ft.eval()
# Run test set
print("Test results:")
test_result(dataloaders, dataset_sizes,
            class_names, criterion, device, model_ft)
