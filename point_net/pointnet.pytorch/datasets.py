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
    def __init__(self, root, npoints = 2500, classification = False, class_choice = None, train = True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}

        self.classification = classification

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        print("cat: ",self.cat)
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        print("cat after class_choice: ",self.cat)
        time.sleep(10)
        # cat:  {'Earphone': '03261776', 'Motorbike': '03790512', 'Rocket': '04099429', 'Car': '02958343', 'Laptop': '03642806', 'Cap': '02954340', 'Skateboard': '04225987', 'Mug': '03797390', 'Guitar': '03467517', 'Bag': '02773838', 'Lamp': '03636649', 'Table': '04379243', 'Airplane': '02691156', 'Pistol': '03948459', 'Chair': '03001627', 'Knife': '03624134'}
        # cat after class_choice:  {'Chair': '03001627'}

        print('##############11')
        self.meta = {}
        for item in self.cat:
            print('category ', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            print(dir_point, dir_seg)
            fns = sorted(os.listdir(dir_point))
#            print('fns ', fns)
            if train:
                fns = fns[:int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9):]
#            print('fns after train/test ', fns)
            #print(os.path.basename(fns))
            time.sleep(2)
            for fn in fns:
#                print('fn ', fn)
                token = (os.path.splitext(os.path.basename(fn))[0])
#                print('token ', token)
#                print('append ', os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg'))
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
        time.sleep(2)

        print('##############22')
        self.datapath = []
        for item in self.cat:
            print('item ', item)
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))
        print('datapath ', self.datapath)
        # [('Chair', 'shapenetcore_partanno_segmentation_benchmark_v0/03001627/points/e3a838cb224367c59fce07ae6c046b8c.pts', 
        # 'shapenetcore_partanno_segmentation_benchmark_v0/03001627/points_label/e3a838cb224367c59fce07ae6c046b8c.seg'), 
        # ('Chair', 'shapenetcore_partanno_segmentation_benchmark_v0/03001627/points/e3b04359a3e9ac5de5dbcc9343304f4a.pts', 
        # 'shapenetcore_partanno_segmentation_benchmark_v0/03001627/points_label/e3b04359a3e9ac5de5dbcc9343304f4a.seg')...]

        print('##############33')
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        print('classes ', self.classes)
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath)/50):
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
#                print('l ', l)
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
        print(self.num_seg_classes)


    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        print("###########get")
        print("point_set shape: ",point_set.shape)
        print("seg shape: ",seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        print("###########get choice")
        print("point_set shape: ",point_set.shape)
        print("seg shape: ",seg.shape)
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

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
