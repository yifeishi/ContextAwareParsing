import os
import sys
import scipy.io as sio
import numpy as np
from scipy.io import loadmat

# read category 10
category_data = loadmat('bedroom_room_feature_test.mat')['category']
print(category_data)

object_count = np.zeros(100)
print(object_count)
# traverse and check the id, count 10
for i in range(0,category_data.shape[0]):
    for j in range(0,category_data.shape[1]):
        if category_data[i][j] != -1:
#            print(category_data[i][j])
            object_count[int(category_data[i][j])] = object_count[int(category_data[i][j])] + 1
print(object_count)

# read category_index_map.txt 5
train_file = open('category_index_map.txt')
training_set = {'dummy': -2}
while 1:
    line = train_file.readline()
    if not line:
        break
    L = line.split(' ')
    training_set[int(L[1])] = L[0]
    
print('..................')
for i in range(0,91):
    if int(object_count[i]) > int(category_data.shape[1]*0.2):     
        print('%s  %d' %(training_set[int(i)],int(object_count[i])))

