#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import scipy.io as sio
import numpy as np

def ObbFeatureTransformer(obj_obb_fea_tmp):
    obj_obb_fea = np.zeros(6)
    obj_obb_fea[0] = (obj_obb_fea_tmp[0] + obj_obb_fea_tmp[3])*0.5
    obj_obb_fea[1] = (obj_obb_fea_tmp[1] + obj_obb_fea_tmp[4])*0.5
    obj_obb_fea[2] = (obj_obb_fea_tmp[2] + obj_obb_fea_tmp[5])*0.5
    obj_obb_fea[3] = obj_obb_fea_tmp[0] - obj_obb_fea_tmp[3]
    obj_obb_fea[4] = obj_obb_fea_tmp[1] - obj_obb_fea_tmp[4]
    obj_obb_fea[5] = obj_obb_fea_tmp[2] - obj_obb_fea_tmp[5]
    return obj_obb_fea


dataNum = 100
maxBoxes = 300
featureSize = 2054
mvFeatureSize = 2048
boxes = np.zeros((featureSize,maxBoxes*dataNum))

# load room txt
g_room_type = 'office'
g_room_file_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/'+g_room_type+'_room.txt'
g_out_file_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/'+g_room_type+'_room_feature'
g_house_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/house'
g_object_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/object'

count = 0
room_file = open(g_room_file_path)
while 1:
    line = room_file.readline()
    if not line:
        break
    L = line.split(' ')
    house_name = L[0]
    room_id = L[1]
    print(house_name)
    print(room_id)
    room_fea = np.zeros((featureSize,maxBoxes))
    countT = 0
    obb_file_path = os.path.join(g_house_path,house_name,'obb_'+room_id+'.txt')
    obb_file = open(obb_file_path)
    while 1:
        lineT = obb_file.readline()
        if not lineT:
            break
        LT = lineT.split(' ')
        obj_name = LT[0]
        LT = LT[1:len(LT)-1]
        obj_obb_fea = list(map(float, LT))
        obj_obb_fea = np.array(obj_obb_fea)
        obj_obb_fea = ObbFeatureTransformer(obj_obb_fea)
        obj_mv_fea_file_path = os.path.join(g_object_path,obj_name,'rgb_img/feature.txt')
        obj_mv_fea_file = open(obj_mv_fea_file_path)
        obj_mv_fea_sum = np.zeros((mvFeatureSize,5))
        obj_mv_fea_mp = np.zeros(mvFeatureSize)
        countTT = 0
        while 1:
            lineTT = obj_mv_fea_file.readline()
            if not lineTT:
                break
            LTT = lineTT.split('  ')
            LTT = LTT[0:len(LTT)]
            obj_mv_fea = list(map(float, LTT))
            obj_mv_fea = np.array(obj_mv_fea)
            obj_mv_fea_sum[:,countTT] = obj_mv_fea
            countTT = countTT + 1
        for i in range(0,mvFeatureSize):
            entry_mp = -99999999
            for j in range(0,5):
                if entry_mp < obj_mv_fea_sum[i,j]:
                    entry_mp = obj_mv_fea_sum[i,j]
            obj_mv_fea_mp[i] = entry_mp
        obj_fea = np.concatenate((obj_mv_fea_mp,obj_obb_fea), axis=0)
        room_fea[:,countT] = obj_fea
        countT = countT + 1
#    print('box num')
#    print(countT)
    boxes[:,count*maxBoxes:count*maxBoxes+maxBoxes] = room_fea
    count = count + 1
    if dataNum == count:
        break

#np.savetxt(g_out_file_path,boxes,fmt="%.3f")  
sio.savemat(g_out_file_path, mdict={'boxes': boxes}, do_compression=True)