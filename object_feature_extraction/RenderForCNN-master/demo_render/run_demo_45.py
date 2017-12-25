#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
RENDERING PIPELINE DEMO
run it several times to see random images with different lighting conditions,
viewpoints, truncations and backgrounds.
'''

import os
import sys
from PIL import Image
import random
from datetime import datetime
random.seed(datetime.now())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'../'))
from global_variables import *

# set debug mode
debug_mode = 1

if debug_mode:
    io_redirect = ''
else:
    io_redirect = ' > /dev/null 2>&1'

# -------------------------------------------
# RENDER
# -------------------------------------------

# define views
viewpoint_samples_file = os.path.join(BASE_DIR, 'sample_viewpoints_45.txt')
viewpoint_samples = [[float(x) for x in line.rstrip().split(' ')] for line in open(viewpoint_samples_file,'r')]
#v = viewpoint_samples[1]

# search the .obj
suncg_sub_dir = os.listdir(g_suncg_object_path)
for obj_dir in suncg_sub_dir:
    file_names = os.listdir(os.path.join(g_suncg_object_path,obj_dir))
    for file_name in file_names:
        L = file_name.split('_')
        if os.path.splitext(file_name)[1] == '.obj' and L[0] == 'unified':
            file_path = os.path.join(g_suncg_object_path,obj_dir,file_name)
            img_folder = os.path.join(g_suncg_object_path,obj_dir,'rgb_img')
            if not os.path.exists(img_folder):
                os.mkdir(img_folder)
            count = 5
            for v in viewpoint_samples:
                image_path = os.path.join(img_folder,str(count)+'.png')
                python_cmd = 'python %s -a %s -e %s -t %s -d %s -m %s -o %s' % (os.path.join(BASE_DIR, 'render_class_view.py'), 
                    str(v[0]), str(v[1]), str(v[2]), str(v[3]), file_path, image_path)
                os.system('%s %s' % (python_cmd, io_redirect))
                count = count + 1


"""
# tmp
print(os.path.join(g_suncg_object_path,'42'))
file_names = os.listdir(os.path.join(g_suncg_object_path,'42'))
print(file_names)

for file_name in file_names:
    strlist = file_name.split('_')
    if os.path.splitext(file_name)[1] == '.obj' and strlist[0] == 'unified':
        print('strlist')
        print(strlist[0])
        print('\n')
        print('file_name')
        print(file_name)
        print('\n')
        file_path = os.path.join(g_suncg_object_path,'42',file_name)
        img_folder = os.path.join(g_suncg_object_path,'42','rgb_img')
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)
        image_path = os.path.join(img_folder,'0.png')
        python_cmd = 'python %s -a %s -e %s -t %s -d %s -m %s -o %s' % (os.path.join(BASE_DIR, 'render_class_view.py'), 
            str(v[0]), str(v[1]), str(v[2]), str(v[3]), file_path, image_path)
        os.system('%s %s' % (python_cmd, io_redirect))
#        print(file_path)
#        print(image_path)
#        print('\n')
"""

"""
# set filepath
syn_images_folder = os.path.join(BASE_DIR, 'demo_images')
model_name = 'chair001'
image_name = 'demo_img.png'
if not os.path.exists(syn_images_folder):
    os.mkdir(syn_images_folder)
    os.mkdir(os.path.join(syn_images_folder, model_name))

print ">> Selected view: ", v
python_cmd = 'python %s -a %s -e %s -t %s -d %s -o %s' % (os.path.join(BASE_DIR, 'render_class_view.py'), 
    str(v[0]), str(v[1]), str(v[2]), str(v[3]), os.path.join(syn_images_folder, model_name, image_name))
#print ">> Running rendering command: \n \t %s" % (python_cmd)
print ">> Running rendering command: \n \t"
os.system('%s %s' % (python_cmd, io_redirect))

"""