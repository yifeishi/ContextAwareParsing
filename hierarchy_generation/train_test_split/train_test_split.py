import os
import random

g_suncg_house_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/house'
g_train_txt_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/train_test_split/SUNCG_train_sceneId_new.txt'
g_test_txt_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/train_test_split/SUNCG_test_sceneId_new.txt'

file_train = open(g_train_txt_path, 'w')
file_test = open(g_test_txt_path, 'w')

suncg_sub_dir = os.listdir(g_suncg_house_path)
for house_dir in suncg_sub_dir:
#    house_path = os.path.join(g_suncg_house_path,house_dir,'house.json')
    ranInt = random.randint(0,9)
#    print(ranInt)
    if ranInt > 1:
        print('%s   train\n'%house_dir)
        file_train.write('%s\n'%house_dir)
    else:
        print('%s          test\n'%house_dir)
        file_test.write('%s\n'%house_dir)

file_train.close()
file_test.close()