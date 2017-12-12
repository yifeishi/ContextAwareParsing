#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys


# suncg path
g_suncg_house_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/house'
g_suncg_category_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/github/ContextAwareParsing/metadata/ModelCategoryMapping.csv'
scn2obb_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/github/ContextAwareParsing/gaps/bin/x86_64/scn2obb'


# search the .obj
suncg_sub_dir = os.listdir(g_suncg_house_path)
for obj_dir in suncg_sub_dir:
    print(g_suncg_house_path)
    print(obj_dir)
    file_names = os.listdir(os.path.join(g_suncg_house_path,obj_dir))
    print(file_names)
    for file_name in file_names:
        if os.path.splitext(file_name).__len__<1:
            continue;
        if os.path.splitext(file_name)[1] == '.json':
            file_path = os.path.join(g_suncg_house_path,obj_dir,file_name)
            obb_file_path = os.path.join(g_suncg_house_path,obj_dir) + '/obb.txt'
            os.getcwd()
            os.chdir('%s' %(os.path.join(g_suncg_house_path,obj_dir)))
            pwd_cmd = 'pwd'
            os.system('%s' % (pwd_cmd))
            gaps_cmd = '%s %s -categories %s -obb %s' %(scn2obb_path, file_path, g_suncg_category_path, obb_file_path)
            os.system('%s' % (gaps_cmd))
            print('\n')
