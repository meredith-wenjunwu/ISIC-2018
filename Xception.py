# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
# import torchvision
# import matplotlib.pyplot as plt
import time
import copy
import numpy as np
# import pretrainedmodels
import pickle
from torch.autograd import Variable
import sklearn.metrics as metrics


def train_model(dataloaders, device, dataset_sizes, model,
                criterion, optimizer, scheduler, output_filename,
                results_dict, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            y_true = []
            y_pred = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                y_true.extend(labels.data.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(
                y_true, y_pred)
            acc = running_corrects.double() / dataset_sizes[phase]

            if (phase == 'train'):
                result_dict = addResult(results_dict, epoch_loss, acc,
                   precision,
                          recall, f1, train=True)
            else:
                result_dict = addResult(results_dict, loss, acc, precision,
                   recall, f1,
                          valid=True)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    with open(output_filename, 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def addResult(result_dict, loss, acc, precision, recall, f1, train=False,
              valid=False, test=False):
    if train:
        result_dict['train']['loss'].append(loss)
        result_dict['train']['acc'].append(acc)
        # print(result_dict)
        result_dict['train']['prec'].append(precision)
        result_dict['train']['recall'].append(recall)
        result_dict['train']['f1'].append(f1)
    elif valid:
        result_dict['valid']['loss'].append(loss)
        result_dict['valid']['acc'].append(acc)
        result_dict['valid']['prec'].append(precision)
        result_dict['valid']['recall'].append(recall)
        result_dict['valid']['f1'].append(f1)
    elif test:
        result_dict['test']['loss'].append(loss)
        result_dict['test']['acc'].append(acc)
        result_dict['test']['prec'].append(precision)
        result_dict['test']['recall'].append(recall)
        result_dict['test']['f1'].append(f1)
    return result_dict


def test_result(results_dict, dataloaders, dataset_sizes, device,
                class_names, criterion, model):
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    running_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy().tolist())
            loss = criterion(outputs, labels)
            y_true.extend(labels.cpu().numpy().tolist())
            running_loss += loss.item() * inputs.size(0)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(len(class_names)):
        print('Accuracy of %5s : %4f%%' % (
            class_names[i], 100 * class_correct[i] / class_total[i]))
    print('Total accuracy: %4f%%' %
          (100 * np.sum(class_correct) / np.sum(class_total)))
    print('Test loss: ', running_loss / dataset_sizes['test'])
    acc = metrics.accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true,
        y_pred, average='weighted')
    print('Precision: %4f%%' % (precision))
    print('Recall: %4f%%' % (recall))
    print('F1: %4f%%' % (f1))
    result_dict = addResult(results_dict, running_loss / dataset_sizes['test'],
              acc, precision, recall, f1, test=True)
    return result_dict
