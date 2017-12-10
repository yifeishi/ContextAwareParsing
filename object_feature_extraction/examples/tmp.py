import argparse
import os
import shutil
import time
import torchvision.models as models
import sys
sys.path.append('/data/05/deepfusion/users/yifeis/sceneparsing/github/ContextAwareParsing/object_feature_extraction') # if needed
import pretrainedmodels
import torch
import pretrainedmodels.utils as utils


#resnet50 = models.resnet50(pretrained='imagenet')

model_names = sorted(name for name in pretrainedmodels.__dict__
    if not name.startswith("__")
    and name.islower()
    and callable(pretrainedmodels.__dict__[name]))
print(model_names)

model_name = 'fbresnet152' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name]()
model.eval()

load_img = utils.LoadImage()

# transformations depending on the model
# rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
tf_img = utils.TransformImage(model) 

path_img = 'data/cat.jpg'

input_img = load_img(path_img)

input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_tensor,
    requires_grad=False)

output_features = model.features(input) # 1x14x14x2048 size may differ
output_logits = model.logits(output_features) # 1x1000

print("output_features:")
print(output_features)

print("output_logits:")
print(output_logits)