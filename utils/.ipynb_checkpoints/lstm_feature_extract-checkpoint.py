'''
Extract 256 dimension LSTM features for ActionSet
'''
from os import listdir
import csv
import sys
import itertools
import random
import torch.nn as nn
from random import shuffle
from torchvision.transforms import functional
from PIL import Image
import config.GTEA as DATA
from torch.autograd import Variable
import torchvision
import torch
from torchvision import datasets, models, transforms
sys.path.append('.')
import numpy as np

num_classes = DATA.rgb_lstm['num_classes']
mean = DATA.rgb_sim['mean']
std = DATA.rgb_sim['std']
png_dir = DATA.rgb_sim['png_dir']
class_map = DATA.rgb_lstm['class_map']
label_dir = DATA.rgb['data_dir'] + DATA.rgb['label_dir']
label_files = listdir(label_dir)
out_dir = DATA.rgb['data_dir']
weights_dir = DATA.rgb_sim['weights_dir']
features_2048_dir = DATA.rgb_sim['features_2048_dir']
lstm_512_features = DATA.rgb_lstm['lstm_512_features']
png_dir = DATA.rgb_sim['png_dir']
data_dir = DATA.rgb_sim['data_dir']
sequence_length = DATA.rgb_lstm['sequence_length']
DEVICE = 1

print ("###################### running testing  #########################")
window = 11
stride = 1
print ("############ Window Size = {} ############ Stride = {} ##############".format(window, stride))

def rgb_sequence_loader(paths, mean, std, inp_size, rand_crop_size, resize_size):
    irand = random.randint(0, inp_size[0] - rand_crop_size[0])
    jrand = random.randint(0, inp_size[1] - rand_crop_size[1])
    flip = random.random()
    batch = []
    for path in paths:
        img = Image.open(path)
        img = img.convert('RGB')
        img = functional.center_crop(img, (inp_size[0], inp_size[1]))
        img = functional.resize(img, [resize_size, resize_size])
        tensor = functional.to_tensor(img)
        tensor = functional.normalize(tensor, mean, std)
        batch.append(tensor)

    batch = torch.stack(batch)

    return batch

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

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, inp):
        x = self.rnn(inp)[0]
        outputs = self.out(x)
        outputs_mean = Variable(torch.zeros(outputs.size()[0], num_classes)).cuda(DEVICE)
        for i in range(outputs.size()[0]):
            outputs_mean[i] = outputs[i].mean(dim=0)

        return outputs_mean

class Net(nn.Module):
    def __init__(self, model_conv, input_size, hidden_size):
        super(Net, self).__init__()
        self.resnet50Bottom = ResNet50Bottom(model_conv)
        self.lstmNet = LSTMNet(input_size, hidden_size)

    def forward(self, inp):
        features = []

        batch_size = inp.size()[0]
        sequence_length = inp.size()[1]
        inp = inp.view(-1, 3, 300, 300)
        for i in range(0, inp.size()[0], 128):
            features.append(self.resnet50Bottom(inp[i:i+128]))

        feature_sequence = torch.cat(features, dim=0)
        feature_sequence = feature_sequence.view(batch_size, sequence_length, 2048)
        outputs = self.lstmNet(feature_sequence)
        return outputs.view(num_classes)


#Loading Pretrained Model
model=torch.load(weights_dir+'weights_rgb_lstm_lr_0.001_momentum_0.9_step_size_20_gamma_1_seq_length_11_num_classes_11_batch_size_128.pt')
model = model.cuda(DEVICE)
print label_files
 #Testing Code
for label in label_files:
    if 'S4' not in label:
        print (label)
        r = csv.reader(open(label_dir + label, 'r'), delimiter = ' ')
        npy_video=[]
        csv_write = csv.writer(open(data_dir +lstm_512_features + '/gt/' + label[:-4] + '.txt','wb'))
        write_trans = open(data_dir +lstm_512_features + '/transcripts/' + label[:-4] + '.txt','wb')
        tot_rows=[]
        for row in r:
            if len(row) > 1:
                tot_rows.append(row)
        r = csv.reader(open(label_dir + label, 'r'), delimiter = ' ')
        for row in r:
            print row
            if len(row) > 1:
                label_str = row[0]
                write_trans.write(format(label_str))
                write_trans.write('\n')
                for i in range(int(row[2]), int(row[3])-window+2, stride):
                    print ([data_dir+ png_dir + label[:-4] + '/' + str(item).zfill(10) + '.png' for item in range(i,i + window)])
                    path = [data_dir+ png_dir + label[:-4] + '/' + str(item).zfill(10) + '.png' for item in range(i,i + window)]
                    sequence = rgb_sequence_loader(path, mean, std, [280, 450], [224, 224], 300)
                    sequence = sequence.cuda(DEVICE)
                    sequence = sequence.view(1, window, 3, 300, 300)
                    sequence = model(sequence)
                    npy_video.append(sequence.cpu().detach().numpy())
                    #print ([[np.asarray(sequence.cpu().detach().numpy()[0])], [int(class_map[row[0]])]])
                    #_, out = torch.max(sequence.data, 1)
                    #print ([str(int(out.cpu().numpy())), str(class_map[row[0]])])
                    #print (str(class_map[row[0]]))
                    csv_write.writerow(str(class_map[row[0]]))

                #npy_video.append(np.asarray([ np.zeros(11), int(class_map[row[0]])]))

                #csv_write.writerow([str(50), str(50)])

        npy_video=np.array(np.transpose(npy_video))
        np.save(data_dir+lstm_512_features + '/features/' + label[:-4] + '.npy', npy_video)
