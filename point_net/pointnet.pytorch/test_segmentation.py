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

colorGallery = np.array([[0, 0, 0], [130, 130, 240], [255, 120, 120], [46, 254, 100], [250, 88, 172],\
[250, 172, 88], [129, 247, 216], [150, 150, 50], [226, 169, 143],\
[8, 138, 41], [1, 223, 215], [11, 76, 95], [190, 182, 90],\
[245, 169, 242], [75, 138, 8], [247, 254, 46], [88, 172, 250 ],\
[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],\
[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],\
[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],\
[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],\
[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],\
[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],\
[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],\
[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[0,0,0],\
[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],\
[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],\
[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[random.randint(0,255),random.randint(0,255),random.randint(0,255)]\
]) 

def writePLY(points, pred_choice, name):
    ###points shape:  (3L, 2500L)
    ###pred_choice shape:  (2500L,)
    ###name: /data/03/yifeis/users/yifeis/sceneparsing/github/ContextAwareParsing/matterport/matterport_data/1LXtFkjw3qL/house_features/region_5/blocks/pts_block_48.txt
    
    L = name.split('/')
    ply_dir = './testply/'+L[len(name.split('/'))-5]+'_'+L[len(name.split('/'))-3]+'_'+L[len(name.split('/'))-2]+'_'+L[len(name.split('/'))-1].split('.')[0]+'.ply'
    print('ply_dir: ', ply_dir)

    output = open(ply_dir, 'w')
    output.write('ply\n')
    output.write('format ascii 1.0\n')
    output.write('element vertex %d\n' %points.shape[1])
    output.write('property float x\n')
    output.write('property float y\n')
    output.write('property float z\n')
    output.write('property uchar red\n')
    output.write('property uchar green\n')
    output.write('property uchar blue\n')
    output.write('property uchar alpha\n')
    output.write('end_header\n')
    for i in range(0,points.shape[1]):
        labelID = int(pred_choice[i])
        output.write('%f %f %f %d %d %d 1\n' %(points[0][i],points[1][i],points[2][i],\
        colorGallery[labelID][0],colorGallery[labelID][1],colorGallery[labelID][2]))
    output.close()



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg',  help='output folder')
parser.add_argument('--model', type=str, default = './seg/seg_model_4.pth',  help='model path')


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

classifier = PointNetDenseCls(k = num_classes)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
classifier.cuda()

num_batch = len(dataset)/opt.batchSize
learning_rate = 0.01

for epoch in range(opt.nepoch):
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate, momentum=0.9)
    for i, data in enumerate(testdataloader, 0):
        points, target, names = data
        
        points, target = Variable(points), Variable(target)
        points = points.transpose(2,1) 
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        pred, _ = classifier(points)
        print("points shape: " ,points.shape)
        print("pred shape1: " ,pred.shape)
        print("target shape1: " ,target.shape)

        # points shape:  (32L, 3L, 2500L)
        # pred shape1:  (32L, 2500L, 41L)
        # target shape1:  (32L, 2500L)

        pred = pred.view(-1, num_classes)
        pred_choice = pred.data.max(1)[1]
        print("pred_choice shape1: " ,pred_choice.shape) 
        pred_choice = pred_choice.view(opt.batchSize, 2500)
        print("pred_choice shape2: " ,pred_choice.shape) 

        for i in range(0, points.shape[0]):
            # pred to pred_choice
            
            # get points: points[i]
            # get pred: pred[i]
            # output ply
            print("###points shape: " ,points[i].shape) 
            print("###pred_choice shape: " ,pred_choice[i].shape)
            print("###name:", names[1][i])
            ####points shape:  (3L, 2500L)
            ###pred_choice shape:  (2500L,)
            ###name: /data/03/yifeis/users/yifeis/sceneparsing/github/ContextAwareParsing/matterport/matterport_data/1LXtFkjw3qL/house_features/region_5/blocks/pts_block_48.txt
            writePLY(points[i], pred_choice[i], names[1][i])

            # get points: points[i]
            # get target: target[i]
            # output gt ply
            print("###points shape: " ,points[i].shape) 
            print("###target shape: " ,target[i].shape)
            print("###name:", names[1][i])
            ####points shape:  (3L, 2500L)
            ###target shape:  (2500L,)
            writePLY(points[i], target[i], names[1][i])

        """
        pred = pred.view(-1, num_classes)
        target = target.view(-1,1)[:,0] - 1
        print("pred shape2: " ,pred.shape) 
        #(batch_size*point_num, 4)
        print("target shape2: " ,target.shape) 
        #(batch_size*point_num)
        """

        invalidPointCount = 0
        for j in range(0,target.shape[0]):
            if target[j].data[0] == 0:
                pred[j][0].data[0] = 0
                pred[j][1:pred.shape[1]].data[0] = -9999
                invalidPointCount = invalidPointCount + 1

        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]

        print("pred_choice shape: " ,pred_choice.shape)

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
        allSample = (float(opt.batchSize * 2500)-invalidPointCount)
        accuracy = correct/allSample
        print('[%d: %d/%d] lr: %f train loss: %f correct: %d all: %d accuracy: %f' %(epoch, i, num_batch, learning_rate, loss.data[0], correct, allSample, accuracy))

    if epoch % 20 == 19:
        learning_rate = learning_rate * 0.5
