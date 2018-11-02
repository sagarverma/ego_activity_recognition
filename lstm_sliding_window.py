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
from utils.folder import NpyPairPreloader, NpySequencePreloader, SequencePreloader
from sklearn.metrics import f1_score
use_gpu = torch.cuda.is_available()
DEVICE = 1

#Data statistics
num_classes = DATA.rgb_sim['num_classes']
class_map = DATA.rgb_sim['class_map']
mean = DATA.rgb_sim['mean']
std = DATA.rgb_sim['std']
features_2048x10x10_dir = DATA.rgb_sim['features_2048x10x10_dir']
features_2048_dir = DATA.rgb_sim['features_2048_dir']
png_dir = DATA.rgb_sim['png_dir']
data_dir = DATA.rgb_sim['data_dir']
weights_dir = DATA.rgb_sim['weights_dir']
train_csv = DATA.rgb_sim['train_csv']
test_csv = DATA.rgb_sim['test_csv']

#Training parameters
lr = DATA.rgb_sim['lr']
momentum = DATA.rgb_sim['momentum']
step_size = DATA.rgb_sim['step_size']
gamma = DATA.rgb_sim['gamma']
num_epochs = DATA.rgb_sim['num_epochs']
batch_size = DATA.rgb_sim['batch_size']

'''
def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def vec_correlation_coefficient(patch1, patch2):
    patch1 = np.reshape(patch1, (patch1.shape[0], patch1.shape[1], -1))
    patch2 = np.reshape(patch2, (patch2.shape[0], patch2.shape[1], -1))

    patch1_mean = patch1.mean(axis=2)
    patch2_mean = patch2.mean(axis=2)

    patch1_mean = np.repeat(patch1_mean[:, :, np.newaxis], patch1.shape[2], axis=2)
    patch2_mean = np.repeat(patch2_mean[:, :, np.newaxis], patch2.shape[2], axis=2)

    product = np.mean((patch1 - patch1_mean) * (patch2 - patch2_mean), axis=2)
    stds = patch1.std(axis = 2) * patch2.std(axis = 2)
    product = np.divide(product, stds, out=np.zeros_like(product), where=stds!=0)
    return product
'''
class ResNet50Bottom(nn.Module):
    """
        Model definition.
    """
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.avg_pool = nn.AvgPool2d(10,1)

    def forward(self, x):
        x = x.view(-1, 3, 300, 300)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(-1, 2048)
        return x

class SimNet(nn.Module):
    """
    Model definition.
    """
    def __init__(self,model_conv, input_size, hidden_size, num_layers, num_classes):
        super(SimNet, self).__init__()
        self.resnet50Bottom = ResNet50Bottom(model_conv)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

    def forward(self, inp, phase):
        if phase == 'train':
            features = []
            #print (inp.size())

            batch_size = inp.size()[0]
            sequence_length = inp.size()[1]

            inp = inp.view(-1, 3, 300, 300)
            for i in range(0, inp.size()[0], 128):
                features.append(self.resnet50Bottom(inp[i:i+128]))

            feature_sequence = torch.cat(features, dim=0)
            feature_sequence = feature_sequence.view(batch_size, sequence_length, 2048)

            # Set initial states
            h0 = torch.zeros(self.num_layers*2, feature_sequence.size(0), self.hidden_size).to(DEVICE) # 2 for bidirection
            c0 = torch.zeros(self.num_layers*2, feature_sequence.size(0), self.hidden_size).to(DEVICE)

            # forward propagate LSTM
            outputs, _ = self.lstm(feature_sequence, (h0, c0))
            outputs = self.fc(outputs[:, -1, :])
            return outputs
        else:
            h0 = torch.zeros(self.num_layers*2, inp.size(0), self.hidden_size).to(DEVICE) # 2 for bidirection
            c0 = torch.zeros(self.num_layers*2, inp.size(0), self.hidden_size).to(DEVICE)
            outputs, _ = self.lstm(inp, (h0, c0))
            outputs = self.fc(outputs[:, -1, :])
            return outputs


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
            gt=[]
            predicted=[]
            for data in dataloaders[phase]:
                inputs, labels = data
                print (inputs.size())
                fg
                if use_gpu:
                    inputs = Variable(inputs.cuda(DEVICE))
                    labels = Variable(labels.cuda(DEVICE))
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs, phase)
                _, preds = torch.max(outputs.data, 1)
                gt.append(labels.cpu().data.numpy())
                predicted.append(preds.cpu().data.numpy())
                #gt_pred=gt_pred.append([preds.cpu().data.numpy(), labels.cpu().data.numpy()])
                #print (gt_pred)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            ''' calculation for F1-score'''
            gt=np.concatenate(gt)
            predicted=np.concatenate(predicted)
            F1_score=f1_score(gt, predicted)



            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                print ('######################################################################################')
                print ("{} loss = {}, acc = {}, F1_score = {},".format(phase, epoch_loss, epoch_acc, F1_score))
            else:
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc)
                print ("{} loss = {}, acc = {}, F1_score = {},".format(phase, epoch_loss, epoch_acc, F1_score))
                print ('######################################################################################')


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
pair_datasets = {'train': SequencePreloader(data_dir + png_dir, data_dir + train_csv, mean, std, [280, 450], [224, 224], 300),
                    'test': NpySequencePreloader(data_dir + features_2048_dir, data_dir + test_csv)}

dataloaders = {x: torch.utils.data.DataLoader(pair_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(pair_datasets[x]) for x in ['train', 'test']}
file_name = __file__.split('/')[-1].split('.')[0]


#Create model and initialize/freeze weights

file_name = __file__.split('/')[-1].split('.')[0]

model_conv = torch.load(weights_dir + 'weights_rgb_cnn_lr_0.001_momentum_0.9_step_size_15_gamma_1_num_classes_10_batch_size_128.pt')
#print(model_conv)
for param in model_conv.parameters():
    param.requires_grad = False

sequence_length = 6
input_size = 2048
hidden_size = 512
num_layers = 1
num_classes = 2
batch_size = 128
num_epochs = 40
model = SimNet(model_conv, input_size, hidden_size, num_layers, num_classes)

if use_gpu:
    model = model.cuda(DEVICE)

#Initialize optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

#Train model
train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)
