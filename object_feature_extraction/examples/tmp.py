import argparse
import os
import shutil
import time
import torchvision.models as models
import sys
sys.path.append('/data/05/deepfusion/users/yifeis/sceneparsing/github/ContextAwareParsing/object_feature_extraction')
import pretrainedmodels
import torch
import pretrainedmodels.utils as utils
import numpy as np


model_name = 'fbresnet152' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name]()
model.cuda()
model.eval()

load_img = utils.LoadImage()
tf_img = utils.TransformImage(model)

g_suncg_object_path = '/data/05/deepfusion/users/yifeis/sceneparsing/data/object'
suncg_sub_dir = os.listdir(g_suncg_object_path)
for obj_dir in suncg_sub_dir:
    for i in range(5,9):
        path_img = os.path.join(g_suncg_object_path,obj_dir,'rgb_img',str(i)+'.png')
        print(path_img)
        input_img = load_img(path_img)
        input_tensor = tf_img(input_img)
        input_tensor = input_tensor.unsqueeze(0)
        input = torch.autograd.Variable(input_tensor.cuda(), requires_grad=False)

        output_features = model.features(input).data.cpu().numpy().squeeze()
        output_features = output_features[:, np.newaxis]
        output_features = output_features.transpose()

        if i == 5:
            output_features_sum = output_features
        else:
            output_features_sum = np.vstack((output_features_sum,output_features))

    np.savetxt(os.path.join(g_suncg_object_path,obj_dir,'rgb_img','feature_new.txt'),output_features_sum, fmt="%f", delimiter="  ")

"""
path_img = '/data/05/deepfusion/users/yifeis/sceneparsing/data/object/43/rgb_img/1.png'
input_img = load_img(path_img)
input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_tensor,
    requires_grad=False)

output_features = model.features(input) # 1x14x14x2048 size may differ
np.savetxt("feature.txt",output_features, fmt="%f", delimiter="  ")

output_logits = model.logits(output_features) # 1x1000

print("output_features:")
print(output_features)

print("output_logits:")
print(output_logits)
"""