from __future__ import print_function, division
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from config import GTEA as DATA
from utils.folder import NpyPairPreloader

use_gpu = torch.cuda.is_available()
DEVICE = 1

#Data statistics
num_classes = DATA.rgb_sim['num_classes']
class_map = DATA.rgb_sim['class_map']

#Training parameters
lr = DATA.rgb_sim['lr']
momentum = DATA.rgb_sim['momentum']
step_size = DATA.rgb_sim['step_size']
gamma = DATA.rgb_sim['gamma']
num_epochs = DATA.rgb_sim['num_epochs']
batch_size = DATA.rgb_sim['batch_size']
features_2048x10x10_dir = DATA.rgb_sim['features_2048x10x10_dir']
data_dir = DATA.rgb_sim['data_dir']
weights_dir = DATA.rgb_sim['weights_dir']
train_csv = DATA.rgb_sim['train_csv']
test_csv = DATA.rgb_sim['test_csv']

class SimNet(nn.Module):
    """
    Model definition.
    """
    def __init__(self, original_model):
        super(SimNet, self).__init__()
        self.fc = nn.Linear(2048, 2)

    def forward(self, xt0, xt1):
        x = F.conv2d(xt0, xt1)
        x = self.fc(x)
        return x

def train_model(model, criterion, optimizer, scheduler, num_epochs=2000):
    """
        Training model with given criterion, optimizer for num_epochs.
    """
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_loss = []
    train_acc = []
    test_acc = []
    test_loss = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                xt1s, xt2s, labels = data
                if use_gpu:
                    xt1s = Variable(xt1s.cuda(DEVICE))
                    xt2s = Variable(xt2s.cuda(DEVICE))
                    labels = Variable(labels.cuda(DEVICE))
                else:
                    xt1s = Variable(xt1s)
                    xt2s = Variable(xt2s)
                    labels = Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                print ('##############################################################')
                print ("{} loss = {}, acc = {},".format(phase, epoch_loss, epoch_acc))
            else:
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc)
                print (" {} loss = {}, acc = {},".format(phase, epoch_loss, epoch_acc))
                print ('##############################################################')


            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model, weights_dir + 'weights_'+ file_name + '_lr_' + str(lr) + '_momentum_' + str(momentum) + '_step_size_' + \
                        str(step_size) + '_gamma_' + str(gamma) + '_num_classes_' + str(num_classes) + \
                        '_batch_size_' + str(batch_size) + '.pt')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))


#Dataload and generator initialization
pair_datasets = {'train': NpyPairPreloader(data_dir + features_2048x10x10_dir, data_dir + train_csv),
                    'test': NpyPairPreloader(data_dir + features_2048x10x10_dir, data_dir + test_csv)}

dataloaders = {x: torch.utils.data.DataLoader(pair_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(pair_datasets[x]) for x in ['train', 'test']}
file_name = __file__.split('/')[-1].split('.')[0]

#Create model and initialize/freeze weights
model = SimNet()

if use_gpu:
    model = model.cuda(DEVICE)

#Initialize optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

#Train model
train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)
