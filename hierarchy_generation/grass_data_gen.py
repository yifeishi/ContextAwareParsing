#!/usr/bin/python
# -*- coding: utf-8 -*-

# input: train_test_split/SUNCG_train_sceneId.txt, room_list/*_room.txt, house/*/hier_*.txt, house/*/obb_*.txt, object/*/obj_img/feature.txt
# output: room_feature/*_room_feature.mat

import os
import sys
import scipy.io as sio
import numpy as np

# define .mat paras
dataNum = 1000
maxBoxes = 100
maxnodes = 100
featureSize = 2054
mvFeatureSize = 2048
boxes_all = np.zeros((featureSize,maxBoxes*dataNum))
op_all = np.zeros((maxnodes,dataNum))
category_all = np.zeros((maxnodes,dataNum))
hier_name_all = ['dummy']

# define path
g_room_type = 'bedroom'
g_room_file_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/room_list/'+g_room_type+'_room.txt'
g_out_file_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/room_feature/'+g_room_type+'_room_feature'
g_suncg_house_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/house'
g_suncg_object_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/object'
g_suncg_class_map_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/github/ContextAwareParsing/metadata/ModelCategoryMapping.csv'
g_train_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/train_test_split/SUNCG_train_sceneId_new.txt'
g_hierarchy_figure_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/hierarchy_visualization/'+g_room_type
if not os.path.exists(g_hierarchy_figure_path):
    os.mkdir(g_hierarchy_figure_path)

def ObbFeatureTransformer(obj_obb_fea_tmp):
    obj_obb_fea = np.zeros(6)
    obj_obb_fea[0] = (obj_obb_fea_tmp[0] + obj_obb_fea_tmp[3])*0.5
    obj_obb_fea[1] = (obj_obb_fea_tmp[1] + obj_obb_fea_tmp[4])*0.5
    obj_obb_fea[2] = (obj_obb_fea_tmp[2] + obj_obb_fea_tmp[5])*0.5
    obj_obb_fea[3] = obj_obb_fea_tmp[0] - obj_obb_fea_tmp[3]
    obj_obb_fea[4] = obj_obb_fea_tmp[1] - obj_obb_fea_tmp[4]
    obj_obb_fea[5] = obj_obb_fea_tmp[2] - obj_obb_fea_tmp[5]
    return obj_obb_fea


def getObjFeature(index, modelid, obb_name):
    # read obb_name, find the obj_id
    obb = open(obb_name)
    obj_fea = np.ones(featureSize) # set floor feature to be all zero
    count = 0
    while 1:
        line = obb.readline()
        if not line:
            break
        L = line.split() 
        if  L[0].split('#')[1] == index and L[1] == modelid:
            count = count + 1
            L = L[3:len(L)]
            # read obb feature
            obj_obb_fea = list(map(float, L))
            obj_obb_fea = np.array(obj_obb_fea)
            obj_obb_fea = ObbFeatureTransformer(obj_obb_fea)
            # read mv image feature
            obj_mv_fea_file_path = os.path.join(g_suncg_object_path,modelid,'rgb_img/feature_new.txt')
            obj_mv_fea_file = open(obj_mv_fea_file_path)
            obj_mv_fea_sum = np.zeros((mvFeatureSize,5))
            obj_mv_fea_mp = np.zeros(mvFeatureSize)
            countTT = 0
            # max pooling
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
    return obj_fea


def genGrassData(hier_name,obb_name,object_class_map,class_index_map):
    boxes = np.zeros((featureSize,maxBoxes))
    op = np.zeros(maxnodes)
    category = np.zeros(maxnodes)
    tmp = np.ones(maxnodes)
    op = op - tmp
    category = category - tmp

    countNode = 0
    countBox = 0
    countLine = 0
    # open hier file
    hier = open(hier_name)
    while 1:
        line = hier.readline()
        if not line:
            break
        L = line.split()
        if countLine == 0:
            # root
            op[countNode] = 1
            category[countNode] = -1
            countNode = countNode + 1
            # floor node
            op[countNode] = 0
            category[countNode] = -1
            countNode = countNode + 1
            obj_fea = getObjFeature(L[1],L[2],obb_name)  #L1 is the local index, L2 is the model id, this is for double check
            boxes[:,countBox] = obj_fea
            countBox = countBox + 1
        elif countLine == 1:
            # first merge node
            op[countNode] = 1
            category[countNode] = -1
            countNode = countNode + 1
        elif countLine > 1 and L[2] != "null":
            # object node
            op[countNode] = 0
            category[countNode] = class_index_map.get(object_class_map.get(L[2],-1),-1)
#            print('\ncategory:')
#            print(class_index_map.get(object_class_map.get(L[2],-1),-1))
#            print(object_class_map.get(L[2],-1),-1)
            countNode = countNode + 1
            obj_fea = getObjFeature(L[1],L[2],obb_name)
            boxes[:,countBox] = obj_fea
            countBox = countBox + 1
            """
            # jerrysyf get class by L[3], add to class_all
            print('\nL[0]')
            print(L[0])
            print('L[1]')
            print(L[1])
            print('L[2]')
            print(L[2])
            print('class')
            print(object_class_map.get(L[2],-1))
            if object_class_map.get(L[2],-1) == -1:
                print('wwwwwwwwwwwwwwwwwwwww')
            """
        else:
            # merge node
            op[countNode] = 1
            countNode = countNode + 1
        countLine = countLine + 1
    return(boxes, op, category)


def getSUNCGClass(g_suncg_class_map_path):
    object_class_map = {'dummy': -1}
    object_class_set = {'dummy': 0}
    class_index_map = {'dummy': -1}
    class_file = open(g_suncg_class_map_path)
    count = 0
    while 1:
        line = class_file.readline()
        if not line:
            break
        L = line.split(',')
        if count > 0:
            object_class_map[L[1]] = L[3]
            object_class_set[L[3]] = 1
        count = count + 1
    count = 0
    for k, v in object_class_set.iteritems():
        class_index_map[k] = count
#        print k, count
        count = count + 1
     # show class_index_map
#    for k, v in class_index_map.iteritems():
#        print k, v
        
    print('.................. has %d object categories '%count)
    return (object_class_map, class_index_map)


# get suncg class
(object_class_map,class_index_map) = getSUNCGClass(g_suncg_class_map_path)

# read room list for training, build map
train_file = open(g_train_path)
training_set = {'dummy': 0}
while 1:
    line = train_file.readline()
    L = line.split('\n')
    training_set[L[0]] = 1 
    if not line:
        break

# load room list for room type
room_file = open(g_room_file_path)
count = 0
while 1:
    line = room_file.readline()
    if not line:
        break
    L = line.split(' ')
    house_id = L[0]
    room_id = L[1]
    # check if it is in the training set
    if(training_set.get(house_id,0) == 1):
        print(house_id)
    else:
        continue
    file_names = os.listdir(os.path.join(g_suncg_house_path,house_id))
    for file_name in file_names:
        # find file named as hier_merge_x.txt
        if file_name.split('_')[0] != 'hier' or file_name.split('.')[len(file_name.split('.'))-1] != 'txt':
            continue
        hier = open(os.path.join(g_suncg_house_path,house_id,file_name))
        hiert = open(os.path.join(g_suncg_house_path,house_id,file_name))
        content1 = hier.readlines()
        line1 = hiert.readline()
         # read the file
        if line1.split(' ')[1] == room_id+'f' and len(content1) > 10 and len(content1) < 100: # line num of hier > 5 and < 100
            hier_name = os.path.join(g_suncg_house_path,house_id,file_name)
            obb_name = os.path.join(g_suncg_house_path,house_id,'obb_'+room_id+'.txt')
            # get grass data for a hier (hier_name--input hier file, obb_name--input obb file, it contains the modelid & obb feature)
            (boxes,op,category) = genGrassData(hier_name,obb_name,object_class_map,class_index_map)
            boxes_all[:,count*maxBoxes:count*maxBoxes+maxBoxes] = boxes
            op_all[:,count] = op
            category_all[:,count] = category
            hier_name_all.append(hier_name)
            count = count + 1
            print('get %d rooms for %s' %(count,g_room_type))
            
            # copy hierarchy figure to folder
            if count > 100:
                continue
            hier_figure_path = os.path.join(g_suncg_house_path,house_id,file_name.split('.')[0]+'.png')
            hier_figure_new_path = os.path.join(g_hierarchy_figure_path,str(count)+'_'+file_name.split('.')[0]+'.png')
            gaps_cmd = 'cp %s %s' %(hier_figure_path, hier_figure_new_path)
            print('copy hierarchy figure')
#            print(gaps_cmd)
            os.system('%s' % (gaps_cmd))
            break
    if count >= dataNum:
        break

hier_name_all.remove('dummy')
sio.savemat(g_out_file_path, mdict={'boxes': boxes_all,'ops': op_all, 'hier_name':hier_name_all, 'category':category_all}, do_compression=True)


# old code
"""
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

"""