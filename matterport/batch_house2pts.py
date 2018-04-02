import os
import time
from time import gmtime, strftime

gaps_path = '/data/03/yifeis/users/yifeis/sceneparsing/github/ContextAwareParsing/matterport/gaps/bin/x86_64/mpview'
data_path = '/n/fs/rgbd/data/matterport/v1'
matterport_path = '/data/03/yifeis/users/yifeis/sceneparsing/github/ContextAwareParsing/matterport/matterport_data'

house_names = os.listdir(data_path)
for house_name in house_names:
    print(house_name)
    house_path = os.path.join(data_path, house_name, 'house_segmentations', house_name + '.house')
    mesh_path = os.path.join(data_path, house_name, 'house_segmentations', house_name + '.ply')
    output_path = os.path.join(matterport_path, house_name)
    gaps_cmd = '%s %s -input_mesh %s -output_object %s' %(gaps_path, house_path, mesh_path, output_path)
    print(gaps_cmd)
    os.system('%s' % (gaps_cmd))