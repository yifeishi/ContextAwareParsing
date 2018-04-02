from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets_matterport import PartDataset
from pointnet import PointNetDenseCls
import torch.nn.functional as F
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')


opt = parser.parse_args()
print (opt)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = False, class_choice = ['Chair'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = False, class_choice = ['Chair'], train = False)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

if os.path.exists('./debug1'):
    shutil.rmtree('./debug1')
if not os.path.exists('./debug1'):
    os.mkdir('./debug1')

blue = lambda x:'\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls(k = num_classes)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
classifier.cuda()

num_batch = len(dataset)/opt.batchSize
learning_rate = 0.01

for epoch in range(opt.nepoch):
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate, momentum=0.9)
    for i, data in enumerate(dataloader, 0):
        points, target, names = data
#        print("points shape: " ,points.shape) #(batch_size, point_num, 3(xyz))
#        print("target shape: " ,target.shape) #(batch_size, point_num, label)
        points, target = Variable(points), Variable(target)
        points = points.transpose(2,1) 
        points, target = points.cuda(), target.cuda()   
        optimizer.zero_grad()
        pred, _ = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1,1)[:,0] - 1
        
#       print("pred shape: " ,pred.shape) #(batch_size*point_num, 4)
#       print("target shape: " ,target.shape) #(batch_size*point_num)
        
        invalidPointCount = 0
        for j in range(0,target.shape[0]):
            if target[j].data[0] == 0:
                pred[j][0].data[0] = 0
                #for k in range(1,pred.shape[1]):
                pred[j][1:pred.shape[1]].data[0] = -9999
                invalidPointCount = invalidPointCount + 1

        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]

        """
        output_pred = open("./debug1/pred_"+str(epoch)+"_"+str(i)+".txt", 'w')
        output_target = open("./debug1/target_"+str(epoch)+"_"+str(i)+".txt", 'w')
        for j in range(0,target.shape[0]):
            if target[j].data[0] != 0:
                output_pred.write('%d\n' %(int(pred_choice[j])+1))
                output_target.write('%d\n' %(int(target[j][0].data[0])+1))
        output_pred.close()
        output_target.close()
        """
        
#        np.savetxt("./debug/pred_choice_"+str(epoch)+"_"+str(i)+".txt", pred_choice.cpu(), fmt='%d')
#        np.savetxt("./debug/target_"+str(epoch)+"_"+str(i)+".txt", target.data.cpu(), fmt='%d')

        correct = pred_choice.eq(target.data).cpu().sum()
        correct = correct - invalidPointCount
        allSample = (float(32 * 2500)-invalidPointCount)
        accuracy = correct/allSample
        print('[%d: %d/%d] lr: %f train loss: %f correct: %d all: %d accuracy: %f' %(epoch, i, num_batch, learning_rate, loss.data[0], correct, allSample, accuracy))

    if epoch % 20 == 19:
        learning_rate = learning_rate * 0.5

        """
        if i % 10 == 0:
            j, data = enumerate(testdataloader, 0).next()
            points, target = data
            points, target = Variable(points), Variable(target)
            points = points.transpose(2,1) 
            points, target = points.cuda(), target.cuda()   
            pred, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1,1)[:,0] - 1

            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.data[0], correct/float(opt.batchSize * 2500)))
        """
    torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch))