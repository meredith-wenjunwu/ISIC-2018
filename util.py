import torch
import torch.nn as nn
import time
import pickle
import copy
import numpy as np
from torch.autograd import Variable
import sklearn.metrics as metrics


def train(dataloaders, dataset_sizes, class_to_idx, cnn,
          criterion, optimizer, scheduler, output_filename,
          results_dict, nepoch=25, convert=False,
          positive=None, negative=None):
    since = time.time()
    best_model_wts = copy.deepcopy(cnn.state_dict())
    best_acc = 0.0

    for epoch in range(nepoch):
        print('Epoch {}/{}'.format(epoch, nepoch - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                cnn.train()  # Set model to training mode
            else:
                cnn.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            y_true = []
            y_pred = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if (convert_label):
                    inputs, labels = convert_label(inputs, labels,
                                                   class_to_idx,
                                                   positive, negative)
                # print(len(inputs))
                # print(len(labels))
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = cnn(inputs)
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
                # print("y_pred")
                # print(y_pred)
                # print("y_true")
                # print(y_true)

            epoch_loss = running_loss / dataset_sizes[phase]
            acc = running_corrects.double() / dataset_sizes[phase]
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(
                y_true, y_pred)
            if (phase == 'train'):
                addResult(results_dict, epoch_loss, acc, precision,
                          recall, f1, train=True)
            else:
                addResult(results_dict, loss, acc, precision, recall, f1,
                          valid=True)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, acc))

            # deep copy the model
            if phase == 'val' and acc > best_acc:
                best_acc = acc
                best_model_wts = copy.deepcopy(cnn.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    with open(output_filename, 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # load best model weights
    cnn.load_state_dict(best_model_wts)
    # torch.save(cnn, weights_dir)
    return cnn, results_dict


def test_result(results_dict, dataloaders, dataset_sizes, class_to_idx,
                class_names, criterion, model, convert=False,
                positive=None, negative=None):
    if (not convert):
        class_correct = list(0. for i in range(len(class_names)))
        class_total = list(0. for i in range(len(class_names)))
    else:
        class_correct = [0] * 2
        class_total = [0] * 2
    running_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            if (convert_label):
                    inputs, labels = convert_label(inputs, labels,
                                                   class_to_idx,
                                                   positive, negative)
            if (len(inputs) > 0):
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()
                # print(inputs.shape)
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
    if (convert):
        print('Accuracy of positive class {} : {}'.format(positive
                , 100 * class_correct[0] / class_total[0]))
        print('Accuracy of negative class {} : {}'.format(negative
                , 100 * class_correct[1] / class_total[1]))
    else:
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
    addResult(results_dict, running_loss / dataset_sizes['test'],
              acc, precision, recall, f1, test=True)


def convert_label(inputs, labels, class_to_idx,
                  newclass_dict):
    # positive - list of classes (string) that will have positive labels
    # negative - list of classes (string) that will have negative labels

    # of classes
    num_classes = len(newclass_dict)
    mapping = {}

    for 
    if len(positive) > 0:
        pos_idx = []
        for p in positive:
            pos_idx.append(class_to_idx[p])
    if len(negative) > 0:
        neg_idx = []
        for n in negative:
            neg_idx.append(class_to_idx[n])
    # print("positive classes: ")
    # print(pos_idx)
    # print("negative classes: ")
    # print(neg_idx)
    pos_mapping = []
    neg_mapping = []
    for i in range(len(inputs)):
        l = labels[i].item()
        if l in pos_idx:
            pos_mapping.append(i)
            # print("label: ")
            # print(l.item())
            # print("positive class + " + str(i))
        elif l in neg_idx:
            neg_mapping.append(i)
            # print("label: ")
            # print(l.item())
            # print("negative class + " + str(i))
    # print(pos_mapping)
    # print(neg_mapping)

    index = pos_mapping.copy()
    index.extend(neg_mapping)
    # print (index)

    new_inputs = torch.index_select(inputs, 0, torch.LongTensor(index))
    temp = [1] * len(pos_mapping)
    temp.extend([0] * len(neg_mapping))
    new_labels = torch.tensor(temp)
    # print(len(new_inputs))
    # print(len(new_labels))
    return new_inputs, new_labels


def calculate_weights(directory, class_names, class_to_idx,
                      positive=None, negative=None):
    from sklearn import preprocessing
    import os
    if positive and negative:
        pos_samples = 0
        neg_samples = 0
        if len(positive) > 0:
            pos_idx = []
            for p in positive:
                pos_idx.append(class_to_idx[p])
        if len(negative) > 0:
            neg_idx = []
            for n in negative:
                neg_idx.append(class_to_idx[n])
        for folder in class_names:
            contents = os.listdir(os.path.join(directory, 'train', folder))
            if class_to_idx[folder] in pos_idx:
                pos_samples += len(contents)
            elif class_to_idx[folder] in neg_idx:
                neg_samples += len(contents)
        prob_pos = np.float64(pos_samples) / (pos_samples + neg_samples)
        return torch.from_numpy(np.array([prob_pos, 1 - prob_pos])).cuda()

    else:
        total = 0
        samples = [0] * len(class_to_idx)

        for folder in class_names:
            contents = os.listdir(os.path.join(directory, 'train', folder))
            total += len(contents)
            samples[class_to_idx[folder]] = len(contents)
        probs = np.float64(np.array(samples)) / total
        weights = 1.00 / probs
        weights = preprocessing.normalize(np.float64(weights.reshape(-1, 1)), axis=0)
        weights = torch.from_numpy(weights).cuda()
        return weights



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
