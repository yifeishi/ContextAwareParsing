#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

# suncg path
g_suncg_object_path = '/data/05/deepfusion/users/yifeis/sceneparsing/data/object'
scn2scn_path = 'mtl2mtl.py'

# search the .obj
suncg_sub_dir = os.listdir(g_suncg_object_path)
for obj_dir in suncg_sub_dir:
    file_names = os.listdir(os.path.join(g_suncg_object_path,obj_dir))
    for file_name in file_names:
        if os.path.splitext(file_name).__len__<1:
            continue;
        if os.path.splitext(file_name)[1] == '.mtl':
            file_path = os.path.join(g_suncg_object_path,obj_dir,file_name)
            new_file_path = os.path.join(g_suncg_object_path,obj_dir) + '/unified_' + file_name
            print(file_path)
            print(new_file_path)
            python_cmd = 'python %s -i %s -o %s' %(scn2scn_path, file_path, new_file_path)
            print(python_cmd)
            os.system('%s' % (python_cmd))       
            print('\n')


"""
# test
obj_path = ''
scn2scn_path = 'scn2scn.py'

# search the .obj
file_path = '40.obj'
new_file_path = 'unified_40.obj'
python_cmd = 'python %s -i %s -o %s' %(scn2scn_path, file_path, new_file_path)
print(python_cmd)
os.system('%s' % (python_cmd))
print('\n') 
"""