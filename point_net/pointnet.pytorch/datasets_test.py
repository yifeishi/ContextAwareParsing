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


class PartDataset(data.Dataset):
    def __init__(self, root, npoints = 2500, classification = False, class_choice = None, train = True):
        self.npoints = npoints
        self.root = root
        self.datapath = []
        self.outputpath = []

        house_names = os.listdir(self.root)
        house_names.sort()
        for house_name in house_names:
#            print('house_name')
#            print(house_name)
            if house_name == 'metadata':
                continue
            region_names = os.listdir(os.path.join(self.root,house_name,'house_features'))
            region_names.sort()
            for region_name in region_names:
                region_dir = os.path.join(self.root,house_name,'house_features',region_name)
#                print('region_dir')
#                print(region_dir)
                pts_names = os.listdir(region_dir)
                pts_names.sort()
                for pts_name in pts_names:
                    if pts_name.split('.')[1] != 'pts':
                        continue
                    pts_dir = os.path.join(region_dir,pts_name)
                    out_dir = os.path.join(region_dir,pts_name.split('.')[0]+'.pointnet')
#                    print('pts_dir')
#                    print(pts_dir)
#                    print('out_dir')
#                    print(out_dir)
                    self.datapath.append(pts_dir)
                    self.outputpath.append(out_dir)
 
        # check the folders in root
        # go to house_features
        # check the folders
        # go to region_x
        # check *.pts
        # store the *.pts and *.pointnet


    def __getitem__(self, index):
        point_set = np.loadtxt(self.datapath[index]).astype(np.float32)
        out_dir = self.outputpath[index]
        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]
        point_set = torch.from_numpy(point_set)
        print(out_dir)
        return point_set, out_dir

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
