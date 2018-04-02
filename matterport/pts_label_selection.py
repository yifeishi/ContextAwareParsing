import argparse
import os
import random
import numpy as np
import math
import shutil

g_matterport_pts_path = "/data/03/yifeis/users/yifeis/sceneparsing/github/ContextAwareParsing/matterport/matterport_data"
validList = np.array([3,5,7,10,11,15,18])
output = open("pts_label_list.txt", 'w')

def getRowNum(file_name):
    count = 0
    file = open(file_name)
    while 1:
        line = file.readline()
        if not line:
            break
        count = count + 1
    return count

def getObjLabel(file_name):
    file = open(file_name)
    objLabels = np.zeros(getRowNum(file_name))
    count = 0
    while 1:
        line = file.readline()
        if not line:
            break
        L = line.split(" ")
        if(len(L)!=17):
            objLabels[count] = int(0)
        else:
            objLabels[count] = int(L[15])
        count = count + 1
    print('objLabels')
    print(objLabels)
    return objLabels

def getAllObjPTS(object_dirs, region_info_dir):
    objLabels = getObjLabel(region_info_dir)
    obj_count = 0
    g_count = 0
    for object_dir in object_dirs:
        if len(object_dir.split(".")) < 2:
            continue
        if object_dir.split(".")[1] != 'pts':
            continue
        object_name = object_dir.split("/")[len(object_dir.split("/"))-1]
        objectID = int(object_name.split(".")[0].split("_")[1])
        pts = np.zeros((getRowNum(object_dir),3))
        labels = np.zeros(getRowNum(object_dir))
        pts_file = open(object_dir)
        count = 0
        while 1:
            line = pts_file.readline()
            if not line:
                break
            L = line.split()
            pts[count][0] = L[0]
            pts[count][1] = L[1]
            pts[count][2] = L[2]
            labels[count] = objLabels[objectID]
            count = count + 1
        if g_count == 0:
            ptsAll = pts
            labelsAll = labels
        else:
            ptsAll = np.vstack((ptsAll, pts))
            labelsAll = np.hstack((labelsAll, labels))
        g_count = g_count + 1
    return (ptsAll,labelsAll)

def getPTSMaxMin(ptsAll):
    room = np.zeros(6)
    room[0] = room[1] = room[2] = -9999
    room[3] = room[4] = room[5] = 9999
    for i in range(0,ptsAll.shape[0]):
        if room[0] < ptsAll[i][0]:
            room[0] = ptsAll[i][0]
        if room[1] < ptsAll[i][1]:
            room[1] = ptsAll[i][1]
        if room[2] < ptsAll[i][2]:
            room[2] = ptsAll[i][2]
        if room[3] > ptsAll[i][0]:
            room[3] = ptsAll[i][0]
        if room[4] > ptsAll[i][1]:
            room[4] = ptsAll[i][1]
        if room[5] > ptsAll[i][2]:
            room[5] = ptsAll[i][2]
    return room

def splitScene(room, step):
    xNum = int(math.ceil((room[0]-room[3])/step))
    yNum = int(math.ceil((room[1]-room[4])/step))
    zNum = int(math.ceil((room[2]-room[5])/step))
    blockNum = xNum*yNum*zNum
    print("xNum %d, yNum %d, zNum %d, blockNum %d" %(xNum,yNum,zNum,blockNum))
    blocks = np.zeros((int(blockNum),6))
    count = 0
    for i in range(0,xNum):
        for j in range(0,yNum):
            for k in range(0,zNum):
                blocks[count][0] = room[3]+step*(i+1)
                blocks[count][1] = room[4]+step*(j+1)
                blocks[count][2] = room[5]+step*(k+1)
                blocks[count][3] = room[3]+step*i
                blocks[count][4] = room[4]+step*j
                blocks[count][5] = room[5]+step*k
                count = count + 1
    return blocks

def writePLY(ptsBlock, labelsBlock, ply_dir):
    output = open(ply_dir, 'w')
    output.write('ply\n')
    output.write('format ascii 1.0\n')
    output.write('element vertex %d\n' %ptsBlock.shape[0])
    output.write('property float x\n')
    output.write('property float y\n')
    output.write('property float z\n')
    output.write('property uchar red\n')
    output.write('property uchar green\n')
    output.write('property uchar blue\n')
    output.write('property uchar alpha\n')
    output.write('end_header\n')
    for i in range(0,ptsBlock.shape[0]):
        labelID = int(labelsBlock[i])
 #       if labelID != 11:
 #           labelID = 0
        output.write('%f %f %f %d %d %d 1\n' %(ptsBlock[i][0],ptsBlock[i][1],ptsBlock[i][2],\
        colorGallery[labelID][0],colorGallery[labelID][1],colorGallery[labelID][2]))
    output.close()

def writeBlockPTS(ptsAll, labelsAll, blocks, block_dir, blockID):
    ptNum=0
    for i in range(0,ptsAll.shape[0]):
#        print("pts: %f %f %f" %(ptsAll[i][0],ptsAll[i][1],ptsAll[i][2]))
#        print("max: %f %f %f" %(blocks[blockID][0],blocks[blockID][1],blocks[blockID][2]))
#        print("min: %f %f %f" %(blocks[blockID][3],blocks[blockID][4],blocks[blockID][5]))
        if ptsAll[i][0] < blocks[blockID][0] and ptsAll[i][0] > blocks[blockID][3]\
            and ptsAll[i][1] < blocks[blockID][1] and ptsAll[i][1] > blocks[blockID][4]\
            and ptsAll[i][2] < blocks[blockID][2] and ptsAll[i][2] > blocks[blockID][5]:
            ptNum=ptNum+1
    print('blockID %d, ptNum %d, ' %(blockID, ptNum))
    if ptNum < 1:
        return 0
    ptsBlock = np.zeros((ptNum,3))
    labelsBlock = np.zeros(ptNum)
    count=0
    for i in range(0,ptsAll.shape[0]):
        if ptsAll[i][0] < blocks[blockID][0] and ptsAll[i][0] > blocks[blockID][3]\
            and ptsAll[i][1] < blocks[blockID][1] and ptsAll[i][1] > blocks[blockID][4]\
            and ptsAll[i][2] < blocks[blockID][2] and ptsAll[i][2] > blocks[blockID][5]:
            ptsBlock[count][0]=ptsAll[i][0]
            ptsBlock[count][1]=ptsAll[i][1]
            ptsBlock[count][2]=ptsAll[i][2]
            labelsBlock[count]=labelsAll[i]
            count=count+1
    np.savetxt(os.path.join(block_dir,'pts_block_'+str(blockID)+'.txt'), ptsBlock, fmt='%.4f')
    np.savetxt(os.path.join(block_dir,'labels_block_'+str(blockID)+'.txt'), labelsBlock, fmt='%d')
    writePLY(ptsBlock, labelsBlock, os.path.join(block_dir,'pts_block_'+str(blockID)+'.ply'))
    return 0

def splitPTS(ptsAll, labelsAll, block_dir):
    # write all
    np.savetxt(os.path.join(block_dir,'pts_region.txt'), ptsAll, fmt='%.4f')
    np.savetxt(os.path.join(block_dir,'labels_region.txt'), labelsAll, fmt='%d')
    writePLY(ptsAll, labelsAll, os.path.join(block_dir,'pts_region.ply'))

    room = getPTSMaxMin(ptsAll)
    step = 2
    blocks = splitScene(room, step)
    blockID = 0
    for i in range(0, blocks.shape[0]):
        writeBlockPTS(ptsAll, labelsAll, blocks, block_dir, blockID)#
        blockID = blockID + 1
    return 0

def getBlockPTS(region_dir):
    block_dir = os.path.join(region_dir,'blocks')
    region_info_dir = os.path.join(region_dir,'region_obb.txt')
    if os.path.exists(block_dir):
        shutil.rmtree(block_dir)
    if not os.path.exists(block_dir):
        os.mkdir(block_dir)
    object_names = os.listdir(region_dir)
    object_names.sort()
    object_dirs = object_names
    count = 0
    for object_name in object_names:
        object_dirs[count] = os.path.join(region_dir,object_name)
        count = count + 1
    (ptsAll,labelsAll) = getAllObjPTS(object_dirs, region_info_dir)
    splitPTS(ptsAll, labelsAll, block_dir)
    return 0

def isRegionValid(region_dir):
    # read region_obb.txt
    # if has more than 5 objects of the preselected category
    file_name = os.path.join(region_dir,'region_obb.txt')
    file = open(file_name)
    objLabels = np.zeros(getRowNum(file_name))
    validFlag = False
    count = 0
    validCount = 0
    while 1:
        line = file.readline()
        if not line:
            break
        L = line.split(" ")
        if(len(L)!=17):
            objLabels[count] = int(0)
        else:
            objLabels[count] = int(L[15])
        for j in range(0,validList.shape[0]):
            if objLabels[count] == validList[j]:
                validCount = validCount + 1
        count = count + 1
    if validCount >= 5:
        validFlag = True
    return validFlag

def getPtsLabelsDir(region_dir):
    block_names = os.listdir(os.path.join(region_dir,'blocks'))
    block_names.sort()
    for block_name in block_names:
        if block_name.split("_")[0] == "pts" and block_name.split("_")[1] == "block" and block_name.split(".")[1] == "txt" and getRowNum(os.path.join(region_dir,'blocks',block_name)) > 100:
            pts_dir = os.path.join(region_dir,'blocks',block_name)
            labels_dir = os.path.join(region_dir,'blocks',"labels_"+block_name.split("_")[1]+"_"+block_name.split("_")[2])
            output.write('%s %s\n' %(pts_dir, labels_dir))
    return 0

house_names = os.listdir(g_matterport_pts_path)
house_names.sort()
for house_name in house_names:
    house_dir = os.path.join(g_matterport_pts_path,house_name,'house_features')
    region_names = os.listdir(house_dir)
    region_names.sort()
    for region_name in region_names:
        print('house: %s  region: %s' %(house_name,region_name))
        region_dir = os.path.join(house_dir,region_name)
        validFlag = isRegionValid(region_dir)
        print('validFlag ',validFlag)
        if validFlag == True:
            getPtsLabelsDir(region_dir)    
output.close()