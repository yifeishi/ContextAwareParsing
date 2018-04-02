from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
#import progressbar
import sys
import torchvision.transforms as transforms
import argparse
import json
import time

class PartDataset(data.Dataset):
    def __init__(self, root, npoints = 2500, class_num = 41, classification = False, class_choice = None, train = True):
        self.npoints = npoints
        self.root = root
        self.ptslist = os.path.join(self.root, 'pts_label_list.txt')
        self.datapath = []
        self.num_seg_classes = class_num

        with open(self.ptslist, 'r') as f:
            for line in f:
                item = 'dump'
                ls = line.strip().split()
                self.datapath.append((item, ls[0], ls[1]))
#        print('datapath ', self.datapath)


    def __getitem__(self, index):
        fn = self.datapath[index]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
#        print("###########get")
#        print("point_set shape: ",point_set.shape)
#        print("seg shape: ",seg.shape)
#        print(seg)
#        print(len(seg))
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]
        seg = seg[choice]
#        print("seg shape: ",seg.shape,seg.max(),seg.min())
        validList = np.array([3,5,7,10,11,15,18])
#        print("seg shape: ",seg.shape[0])
        
        for i in range(0,seg.shape[0]):
            validFlag = False
            for j in range(0,validList.shape[0]):
                if seg[i] == validList[j]:
                    validFlag = True
            if validFlag == False:
#                print(seg[i])
                seg[i] = 1
#        print("seg unique: ",np.unique(seg))

#        print("###########get choice")
#        print("point_set shape: ",point_set.shape)
#        print("seg shape: ",seg.shape)
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        
        return point_set, seg, fn

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'])
    print(len(d))
    ps, seg = d[0]
    print(ps.size(), ps.type(), seg.size(),seg.type())

    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
    print(len(d))
    ps, cls = d[0]
    print(ps.size(), ps.type(), cls.size(),cls.type())
