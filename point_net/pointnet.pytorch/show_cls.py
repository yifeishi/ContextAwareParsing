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
from datasets_test import PartDataset
from pointnet import PointNetCls
import torch.nn.functional as F
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = '',  help='model path')

opt = parser.parse_args()
print (opt)

classifier = torch.load('./cls/cls_model_0.pkl')
classifier.cuda()
classifier.eval()

test_dataset = PartDataset(root = '/data/03/yifeis/users/yifeis/sceneparsing/github/ContextAwareParsing/matterport/matterport_data' , train = False, classification = True)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle = False)

for i, data in enumerate(testdataloader, 0):
    points, outdir = data    
    points = Variable(points)
    points = points.transpose(2,1)
    points = points.cuda()
    pred, feature, _ = classifier(points)
    print(outdir[0])
    np.savetxt(outdir[0],feature.data.cpu().numpy(), fmt="%f", delimiter="  ")