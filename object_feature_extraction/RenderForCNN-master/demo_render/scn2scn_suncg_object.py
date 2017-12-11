#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
from PIL import Image
import random
from datetime import datetime
random.seed(datetime.now())

# suncg path
g_suncg_object_path = '/data/05/deepfusion/users/yifeis/sceneparsing/data/object'
g_gaps_scn2scn_path = '/data/05/deepfusion/users/yifeis/sceneparsing/github/ContextAwareParsing/gaps/bin/x86_64/scn2scn'

# search the .obj
suncg_sub_dir = os.listdir(g_suncg_object_path)
for obj_dir in suncg_sub_dir:
    file_names = os.listdir(os.path.join(g_suncg_object_path,obj_dir))
    for file_name in file_names:
        if os.path.splitext(file_name)[1] == '.obj':
            file_path = os.path.join(g_suncg_object_path,obj_dir,file_name)
            new_file_path = os.path.join(g_suncg_object_path,obj_dir) + '/unified_' + file_name
            print(file_path)
            print(new_file_path)
            python_cmd = '%s %s %s' %(g_gaps_scn2scn_path, file_path, new_file_path)
            print(python_cmd)
            os.system('%s' % (python_cmd))       
            print('\n') 
