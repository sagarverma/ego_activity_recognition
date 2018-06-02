import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence

import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time, os, csv, shutil, random, itertools
import numpy as np

from rnn_modules import RNN

import torch.backends.cudnn as cudnn

import math

from sklearn.metrics import accuracy_score

cudnn.enabled = False

lr = 0.01
momentum = 0.9
gamma = 1
num_epochs = 50
batch_size = 21

num_classes = 11

batch_norm = False
sequence_length = 2000
recurrent_max = pow(2, 1/sequence_length)

bidirectional = True

class_map = {'x':0, 'bg':0, 'fold':1, 'pour':2, 'put':3, 'scoop':4, 'shake':5, 'spread':6, 'stir':7, 'take':8, 'open': 9, 'close':10}

label_dir = '../../dataset/gtea_labels_cleaned//'
rgb_feature_dir = '../../dataset/rgb_2048_features/'
flow_feature_dir = '../../dataset/flow_2048_features/'

label_files = os.listdir(label_dir)

train_split = [label_file for label_file in label_files if 'S4' not in label_file]
test_split = [label_file for label_file in label_files if 'S4' in label_file]


def load_data(label_file):
    x_rgb = np.load(rgb_feature_dir + label_file[:-4] + '.npy')
    x_flow = np.load(flow_feature_dir + label_file[:-4] + '.npy')
    y = np.zeros((x_rgb.shape[0]))
    
    r = csv.reader(open(label_dir + label_file, 'r'), delimiter=' ')
    for row in r:
        if len(row) > 1:
            for i in range(int(row[2])-1,  int(row[3])):
                y[i] = class_map[row[0]]
    
    #y_act = [k for k,g in itertools.groupby(list(y))]
    
    return x_rgb, x_flow, y

      
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer=1):
        super(Net, self).__init__()
        self.indrnn = RNN(input_size, hidden_size, nonlinearity='indrelu', hidden_max_abs=recurrent_max, batch_first=False, bidirectional=bidirectional)
        
        if bidirectional:
            self.w_att = nn.Linear(hidden_size + hidden_size, 1)
            self.out = nn.Linear(hidden_size + hidden_size, num_classes)
        else:
            self.w_att = nn.Linear(hidden_size, 1)
            self.out = nn.Linear(hidden_size, num_classes)
            
    def forward(self, rgb_inputs, flow_inputs, lengths):
        rgb_inputs = pad_sequence(rgb_inputs, batch_first=False)
        rgb_inputs = pack_padded_sequence(rgb_inputs, lengths, batch_first=False)
        rgb_inputs = rgb_inputs.cuda()
        rgb_outputs, rgb_hiddens = self.indrnn(rgb_inputs)
        rgb_outputs, _ = pad_packed_sequence(rgb_outputs, batch_first=False)
        
        flow_inputs = pad_sequence(flow_inputs, batch_first=False)
        flow_inputs = pack_padded_sequence(flow_inputs, lengths, batch_first=False)
        flow_inputs = flow_inputs.cuda()
        flow_outputs, flow_hiddens = self.indrnn(flow_inputs)
        flow_outputs, _ = pad_packed_sequence(flow_outputs, batch_first=False)
        
        alpha = F.sigmoid(self.w_att(rgb_outputs) + self.w_att(flow_outputs))
        fused_outputs = F.mul(alpha, rgb_outputs) + F.mul((1-alpha), flow_outputs)
        
        outputs = self.out(fused_outputs)
        #outputs = outputs.view(-1, num_classes)
        return outputs

model = Net(2048, 512)        
model = model.cuda()
criterian = nn.CrossEntropyLoss(ignore_index=num_classes)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


for epoch in range(0,num_epochs):
    random.shuffle(train_split)
    
    for phase in ['train', 'test']:
        if phase == 'train':
            model.train(True)
        else:
            model.train(False)
            
        running_loss = 0.0
        running_acc = 0
        
        if phase == 'train':
            for batch_no in range(len(train_split)//batch_size):
                sample = train_split[batch_size * batch_no : batch_size * (batch_no + 1)]
            
                inps = []
                
                sample_no = 0
                for s in sample:
                    data = load_data(s)
                    inps.append([data[0].shape[0], torch.tensor(data[0]), torch.tensor(data[1]), list(data[2])])

                
                inps.sort(key=lambda x: x[0])
                inps.reverse()

                
                rgb_inputs = []
                flow_inputs = []
                output_sizes = []
                for inp in inps:
                    rgb_inputs.append(inp[1])
                    flow_inputs.append(inp[2])
                    output_sizes.append(inp[1].size()[0])
                
                labels = []
                label_sizes = []
                for inp in inps:
                    labels.append(torch.IntTensor(inp[3]))
                    label_sizes.append(len(inp[3]))
                
                optimizer.zero_grad()
                
                outputs = model(rgb_inputs, flow_inputs, output_sizes)
                
                labels = pad_sequence(labels, batch_first=False, padding_value=num_classes)
                labels = labels.cuda()
                
                output_sizes = Variable(torch.IntTensor(output_sizes))
                #label_sizes = Variable(torch.IntTensor(label_sizes))
                
                loss = criterian(outputs.view(-1, num_classes), labels.view(-1).type(torch.cuda.LongTensor))
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                acc = 0
                
                _, predictions = torch.max(outputs.data, 2)
                predictions = predictions.data.cpu().numpy()
                
                labels = labels.data.cpu().numpy()
                
                predictions = np.transpose(predictions, (1,0))
                labels = np.transpose(labels, (1,0))
                

                for i in range(labels.shape[0]):
                    acc += accuracy_score(labels[i][:label_sizes[i]], predictions[i][:label_sizes[i]], normalize=True)
                
                running_acc += acc / labels.shape[0]
                
                
            print (running_loss / int(math.ceil(len(train_split)/batch_size)), running_acc / int(math.ceil(len(train_split)/ batch_size)))