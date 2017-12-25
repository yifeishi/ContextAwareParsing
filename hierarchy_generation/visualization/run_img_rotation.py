#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
from PIL import Image
import random
from datetime import datetime
random.seed(datetime.now())

g_suncg_object_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/object'

# search the .obj
suncg_sub_dir = os.listdir(g_suncg_object_path)
for obj_dir in suncg_sub_dir:
    file_name =os.path.join(g_suncg_object_path,obj_dir,'rgb_img','8.png')
    print(file_name)
    file_out_name = os.path.join(g_suncg_object_path,obj_dir,'rgb_img','8_rotation.png')
    print(file_out_name)
    img = Image.open(file_name)
    out1 = img.rotate(90)
    out1.save(file_out_name)


"""
# tmp
file_name = os.path.join(g_suncg_object_path,'42','rgb_img','8.png')
print(file_name)
file_out_name = os.path.join(g_suncg_object_path,'42','rgb_img','8_rotation.png')
print(file_out_name)
img = Image.open(file_name)
out1 = img.rotate(90)
out1.save(file_out_name)
"""