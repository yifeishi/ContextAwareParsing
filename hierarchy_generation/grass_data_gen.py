#!/usr/bin/python
# -*- coding: utf-8 -*-

# input: train_test_split/SUNCG_train_sceneId.txt, room_list/*_room.txt, house/*/hier_*.txt, house/*/obb_*.txt, object/*/obj_img/feature.txt
# output: room_feature/*_room_feature.mat

import os
import sys
import scipy.io as sio
import numpy as np
import random

# define .mat paras
dataNum = 80000
maxBoxes = 50
matDataNum = 20
samplePerturbateTimes = 50
maxnodes = 50
featureSize = 2060 # 2048+location+size+Xmax+Ymax+Zmax+Xmin+Ymin+Zmin
mvFeatureSize = 2048
train_test = 'train'

# define path
g_room_type = 'bedroom'
g_room_file_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/room_list/'+g_room_type+'_room.txt'
if train_test == 'train':
    g_out_file_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/room_feature/'+g_room_type+'_room_feature'
elif train_test == 'test':
    g_out_file_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/room_feature/'+g_room_type+'_room_feature_test'
g_suncg_house_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/house'
g_suncg_object_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/object'
g_suncg_class_map_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/github/ContextAwareParsing/metadata/ModelCategoryMapping.csv'
if train_test == 'train':
    g_train_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/train_test_split/SUNCG_train_sceneId_new.txt'
elif train_test == 'test':
    g_train_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/train_test_split/SUNCG_test_sceneId_new.txt'
g_hierarchy_figure_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/hierarchy_visualization/'+g_room_type
if not os.path.exists(g_hierarchy_figure_path):
    os.mkdir(g_hierarchy_figure_path)

# xMax,yMax,zMax,xMin,yMin,zMin to size,cen
def ObbFeatureTransformer(obj_obb_fea_tmp):
    obj_obb_fea = np.zeros(6)
    obj_obb_fea[0] = obj_obb_fea_tmp[0] - obj_obb_fea_tmp[3]
    obj_obb_fea[1] = obj_obb_fea_tmp[1] - obj_obb_fea_tmp[4]
    obj_obb_fea[2] = obj_obb_fea_tmp[2] - obj_obb_fea_tmp[5]
    obj_obb_fea[3] = (obj_obb_fea_tmp[0] + obj_obb_fea_tmp[3])*0.5
    obj_obb_fea[4] = (obj_obb_fea_tmp[1] + obj_obb_fea_tmp[4])*0.5
    obj_obb_fea[5] = (obj_obb_fea_tmp[2] + obj_obb_fea_tmp[5])*0.5
    return obj_obb_fea

# size,cen to xMax,yMax,zMax,xMin,yMin,zMin
def ObbFeatureTransformerReverse(obj_obb_fea_tmp):
    obj_obb_fea = np.zeros(6)
    obj_obb_fea[0] = obj_obb_fea_tmp[3] + obj_obb_fea_tmp[0]*0.5
    obj_obb_fea[1] = obj_obb_fea_tmp[4] + obj_obb_fea_tmp[1]*0.5
    obj_obb_fea[2] = obj_obb_fea_tmp[5] + obj_obb_fea_tmp[2]*0.5
    obj_obb_fea[3] = obj_obb_fea_tmp[3] - obj_obb_fea_tmp[0]*0.5
    obj_obb_fea[4] = obj_obb_fea_tmp[4] - obj_obb_fea_tmp[1]*0.5
    obj_obb_fea[5] = obj_obb_fea_tmp[5] - obj_obb_fea_tmp[2]*0.5
    return obj_obb_fea

def getObjFeature(index, modelid, obb_name):
    # read obb_name, find the obj_id
    obb = open(obb_name)
    obj_fea = np.ones(featureSize) # set floor feature to be all zero
    obj_reg = np.ones(6)
    room_cen = np.ones(3)
    count = 0
    while 1:
        line = obb.readline()
        if not line:
            break
        L = line.split()
        ##########################################
        if L[0] == 'Room':
            L = L[2:len(L)]
            # read obb feature
            room_obb_fea = list(map(float, L))
            room_obb_fea = np.array(room_obb_fea)
            room_obb_fea = ObbFeatureTransformer(room_obb_fea)
            room_cen[0] = room_obb_fea[3]
            room_cen[1] = room_obb_fea[4]
            room_cen[2] = room_obb_fea[5]
        ##########################################
        elif  L[0] != 'Room' and L[0].split('#')[1] == index and L[1] == modelid:
            count = count + 1
            L = L[3:len(L)]
            # read obb feature
            obj_obb_fea = list(map(float, L))
            obj_obb_fea = np.array(obj_obb_fea)
            obj_obb_fea_t = ObbFeatureTransformer(obj_obb_fea)
            ##########################################
            # add random noisy to obj_obb_fea
            perturbation = np.zeros(6)
            for i in range(0,3):
#                perturbation[i] = 0
                perturbation[i] = (random.random()-0.5)*obj_obb_fea_t[i]
            for i in range(3,6):
#                perturbation[i] = 0
                perturbation[i] = (random.random()-0.5)*0.1
#            print('............mmmmmmmm')
#            print(perturbation)
#            print('...........')
#            print(obj_obb_fea_t)
            obj_obb_fea_t = obj_obb_fea_t + perturbation
            obj_obb_fea = ObbFeatureTransformerReverse(obj_obb_fea_t)
#            print('...........xxxxxxx')
#            print(obj_obb_fea_t)
            ##########################################
            obj_obb_fea[0] = obj_obb_fea[0] - room_cen[0]
            obj_obb_fea[1] = obj_obb_fea[1] - room_cen[1]
            obj_obb_fea[2] = obj_obb_fea[2] - room_cen[2]
            obj_obb_fea[3] = obj_obb_fea[3] - room_cen[0]
            obj_obb_fea[4] = obj_obb_fea[4] - room_cen[1]
            obj_obb_fea[5] = obj_obb_fea[5] - room_cen[2]
            obj_obb_fea_t[3] = obj_obb_fea_t[3] - room_cen[0]
            obj_obb_fea_t[4] = obj_obb_fea_t[4] - room_cen[1]
            obj_obb_fea_t[5] = obj_obb_fea_t[5] - room_cen[2]
            ##########################################
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
            obj_fea = np.concatenate((obj_mv_fea_mp,obj_obb_fea_t,obj_obb_fea), axis=0)
            obj_reg = -perturbation
    return obj_fea,obj_reg

def genGrassData(hier_name,obb_name,object_class_map,class_index_map):
    boxes = np.zeros((featureSize,maxBoxes))
    boxes_reg = np.zeros((6,maxBoxes))
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
            obj_fea,obj_reg = getObjFeature(L[1],L[2],obb_name)  #L1 is the local index, L2 is the model id, this is for double check
            boxes[:,countBox] = obj_fea
            boxes_reg[:,countBox] = obj_reg
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
            obj_fea,obj_reg = getObjFeature(L[1],L[2],obb_name)
            boxes[:,countBox] = obj_fea
            boxes_reg[:,countBox] = obj_reg
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
    return(boxes, boxes_reg, op, category)


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
    if count%(matDataNum*samplePerturbateTimes) == 0:
        if count != 0:
            g_out_file_mat_path = g_out_file_path + '_' + str(count/matDataNum/samplePerturbateTimes)
            if not os.path.exists(g_out_file_mat_path+'.mat'):
                hier_name_all.remove('dummy')
                sio.savemat(g_out_file_mat_path, mdict={'boxes': boxes_all,'boxes_reg': boxes_reg_all,'ops': op_all, 'hier_name':hier_name_all, 'category':category_all}, do_compression=True)
                print('write to %s' %g_out_file_mat_path)
        boxes_all = np.zeros((featureSize,maxBoxes*matDataNum*samplePerturbateTimes))
        boxes_reg_all = np.zeros((6,maxBoxes*matDataNum*samplePerturbateTimes))
        op_all = np.zeros((maxnodes,matDataNum*samplePerturbateTimes))
        category_all = np.zeros((maxnodes,matDataNum*samplePerturbateTimes))
        count_local = 0
        hier_name_all = ['dummy']
        print('clean data\n')
    
    line = room_file.readline()
    if not line:
        break
    L = line.split(' ')
    house_id = L[0]
    room_id = L[1]
    # check if it is in the training set
    if(training_set.get(house_id,0) == 1):
#        print(house_id)
        xx = 1
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
        if line1.split(' ')[1] == room_id+'f' and len(content1) > 5 and len(content1) < 50: # line num of hier (node num) > 5 and < 50
            hier_name = os.path.join(g_suncg_house_path,house_id,file_name)
            obb_name = os.path.join(g_suncg_house_path,house_id,'obb_'+room_id+'.txt')
            # get grass data for a hier (hier_name--input hier file, obb_name--input obb file, it contains the modelid & obb feature)
            for i in range(0, samplePerturbateTimes):
                (boxes,boxes_reg,op,category) = genGrassData(hier_name,obb_name,object_class_map,class_index_map)
                boxes_all[:,count_local*maxBoxes:count_local*maxBoxes+maxBoxes] = boxes
                boxes_reg_all[:,count_local*maxBoxes:count_local*maxBoxes+maxBoxes] = boxes_reg
                op_all[:,count_local] = op
                category_all[:,count_local] = category
                hier_name_all.append(hier_name)
                count = count + 1
                count_local = count_local + 1
                print('get %d room samples for %s, this is the %dth sample in the %dth mat file, this is the %dth sample for the room' %(count,g_room_type,count_local,(count-1)/matDataNum/samplePerturbateTimes+1,i+1))
            # copy hierarchy figure to folder
            if count > 100:
                continue
            hier_figure_path = os.path.join(g_suncg_house_path,house_id,file_name.split('.')[0]+'.png')
            hier_figure_new_path = os.path.join(g_hierarchy_figure_path,str(count)+'_'+file_name.split('.')[0]+'.png')
            gaps_cmd = 'cp %s %s' %(hier_figure_path, hier_figure_new_path)
#            print('copy hierarchy figure')
#            os.system('%s' % (gaps_cmd))
            break
    if count >= dataNum:
        break