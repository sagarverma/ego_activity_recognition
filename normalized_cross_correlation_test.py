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
import csv
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
test_csv = DATA.rgb_sim['test_sliding_widow_csv']
out_dir = DATA.rgb['data_dir']


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

class SimNet(nn.Module):
    """
    Model definition.
    """
    def __init__(self):
        super(SimNet, self).__init__()
        self.fc = nn.Linear(2048, 2)

    def forward(self, xt0, xt1):
        xt0 = xt0.cpu().numpy()
        xt1 = xt1.cpu().numpy()

        cross_corln = vec_correlation_coefficient(xt0, xt1)

        """
        cross_corln = []
        for i in range(batch_size):
            temp_cross = []
            for j in range(2048):
                patch1 = xt0[i, j, :, :]
                patch2 = xt1[i, j, :, :]
                temp_cross.append(correlation_coefficient((patch1), (patch2)))
            cross_corln.append(np.asarray(temp_cross))

        cross_corln = np.asarray(cross_corln)
        """

        x = torch.from_numpy(cross_corln).type('torch.FloatTensor').cuda(DEVICE)
        x = x.view(-1,2048)
        x = self.fc(x)

        return x

#Dataload and generator initialization
#Create model and initialize/freeze weights
model = SimNet()
model=torch.load(weights_dir+'weights_normalized_cross_correlation_lr_0.001_momentum_0.9_step_size_20_gamma_1_num_classes_2_batch_size_128.pt')
if use_gpu:
    model = model.cuda(DEVICE)
#Initialize optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
#Train model
since = time.time()
test_acc = []
test_loss = []

model.train(False)
r = csv.reader(open(out_dir+test_csv, 'r'), delimiter=',')
output_csv = csv.writer(open(out_dir + 'output_sliding_window.csv','wb'))

vis_npy=[]
for data in r:
    xt1 = np.asarray([np.load(data_dir+features_2048x10x10_dir + data[0])])
    xt2 = np.asarray([np.load(data_dir+features_2048x10x10_dir + data[1])])
    label = np.asarray(int(data[2]))

    xt1 = torch.from_numpy(xt1).type('torch.FloatTensor').cuda(DEVICE)
    xt2 = torch.from_numpy(xt2).type('torch.FloatTensor').cuda(DEVICE)

    if use_gpu:
        xt1 = Variable(xt1.cuda(DEVICE))
        xt2 = Variable(xt2.cuda(DEVICE))
    else:
        xt1 = Variable(xt1)
        xt2 = Variable(xt2)

    output = model(xt1, xt2)
    _, pred = torch.max(output.data, 1)
    print (pred.cpu().data.numpy()[0])
    predicted=pred.cpu().data.numpy()[0]
    output_csv.writerow([data[2], predicted])
