from torchvision import datasets, transforms
import torch.optim as optim
import os
import torch.nn as nn
import torch
import argparse
from torch.autograd import Variable
from torch.optim import lr_scheduler
from cnn_finetune import make_model
# from Xception import *
from util import *
# from torch.autograd import Variable
# from model_withESP import *


# need to find out nromalization param
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((800, 650)),
        transforms.RandomCrop((600, 450)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.765, 0.547, 0.572], [0.142, 0.153, 0.170])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.765, 0.547, 0.572], [0.142, 0.153, 0.170])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.765, 0.547, 0.572], [0.142, 0.153, 0.170])
    ]),
}

parser = argparse.ArgumentParser(description='Dermoscopic image diagnosis')
parser.add_argument('--nepoch', type=int, default=10, help='number of epochs')
# parser.add_argument('--patch_dir', type=str,
#                      default='/Users/wuwenjun/GitHub/EE-511/Project/nuclei_patches',
#                      help='nuclei patches directory')
parser.add_argument('--batch_size', type=int, default=24, help='batch size')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--step_size', type=int, default=5,
                    help='SGD step_size')
# parser.add_argument('--cuda_device', type=int, default=2,
#                     help='GPU number')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--output_filename', type=str, default='result.csv',
                    help='filename of output pickle file')
parser.add_argument('--model_dir', type=str, default='./myModel.pt',
                    help='Model directory and filename')


args = parser.parse_args()
nepoch = args.nepoch
batch_size = args.batch_size
seed = args.seed
lr = args.lr
# device_ind = args.cuda_device
momentum = args.momentum
output_filename = args.output_filename
step_size = args.step_size
model_dir = args.model_dir

# device = torch.device("cuda:{}".format(device_ind)
#                       if torch.cuda.is_available() else "cpu")
data_dir = '/projects/melanoma/ISIC/Meredith/Data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes
# total = 0
# length = []
# for folder in class_names:
#     contents = os.listdir(os.path.join(data_dir, 'train', folder))
#     total += len(contents)
#     length.append(len(contents))
# probs = np.float64(np.array(length)) / total
# weights = 1.00 / probs
# weights = preprocessing.normalize(np.float64(weights.reshape(-1, 1)), axis=0)
# weights = torch.from_numpy(weights).cuda()

weights = calculate_weights(data_dir, class_names,
                            image_datasets['train'].class_to_idx,
                            positive=['BCC', 'BKL', 'DF'],
                            negative=['AKIEC', 'NV', 'MEL'])

# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=batch_size,
                                              shuffle={'train': True, 'val':
                                                       True, 'test': False}[x],
                                              num_workers=8)
               for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes


results = {'train': {'loss': [], 'acc': [], 'prec': [], 'f1': [],
                     'recall': []},
           'valid': {'loss': [], 'acc': [], 'prec': [], 'f1': [],
                     'recall': []},
           'test': {'loss': [], 'acc': [], 'prec': [], 'f1': [],
                    'recall': [], 'loss': []}}

#----------------------------Xception---------------------------

# model_ft = make_model('xception', num_classes=7, pretrained=True,
#                               input_size=(224, 224))

#----------------------------Inception---------------------------
# model_ft = make_model('inception_v4', num_classes=7,
#                       pretrained=True, input_size=(600, 450))
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 7)

#----------------------------ESPNet---------------------------
# model_ft = CNN()
# -----------------------------DenseNet161----------------------
# model_ft = make_model('densenet161', num_classes=2, pretrained=True, input_size=
# (600, 450))
# densenet161 = densenet161.to(device)


# criterion = nn.CrossEntropyLoss()
model_ft = torch.load(model_dir)
model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss(weight=weights.float())

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.8)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                       step_size=step_size, gamma=0.1)

# model_ft, results_dict = train(dataloaders, dataset_sizes,
#                                image_datasets['train'].class_to_idx, model_ft,
#                                criterion, optimizer_ft, exp_lr_scheduler,
#                                output_filename, results, convert=True,
#                                positive = ['BCC',
#                                'BKL', 'DF'], negative = ['AKIEC', 'NV', 'MEL'],
#                                nepoch=nepoch)
# torch.save(model_ft, model_dir)

# model_ft = torch.load(model_dir)

# Run test set
model_ft.eval()
print("Test results:")
result_dict = test_result(results, dataloaders, dataset_sizes,
                          image_datasets['test'].class_to_idx,
                          class_names, criterion, model_ft, convert=True,
                          positive=['BCC', 'BKL', 'DF'],
                          negative=['AKIEC', 'NV', 'MEL'])

with open(output_filename, 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# running_corrects = 0
# loss = 0.0

# # Iterate over data.
# for inputs, labels in dataloaders['test']:
#     torch.cuda.empty_cache()
#     inputs = Variable(inputs).to(device)
#     labels = Variable(labels).to(device)
#     outputs = model_ft(inputs)
#     _, preds = torch.max(outputs, 1)
#     loss += criterion(outputs, labels).item() * inputs.size(0)
#     running_corrects += torch.sum(preds == labels.data)

# test_loss = loss / dataset_sizes['test']
# test_acc = running_corrects.double() / dataset_sizes['test']

# print("test loss: {}".format(test_loss))
# print("test accuracy: {}".format(test_acc))
