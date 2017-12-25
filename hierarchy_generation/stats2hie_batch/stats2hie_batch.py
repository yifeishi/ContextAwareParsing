# Import Tree instance and faces module
import os

g_suncg_house_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/house'
g_suncg_object_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/object'
g_suncg_relation_path = '/n/fs/suncg/planner5d/house_relations'

stats2hie_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/github/ContextAwareParsing/gaps/bin/x86_64/stats2hie'
#house_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/house/0004d52d1aeeb8ae6de39d6bd993e992/house.json'
#stats_path = '/n/fs/suncg/planner5d/house_relations/0004d52d1aeeb8ae6de39d6bd993e992/0004d52d1aeeb8ae6de39d6bd993e992.stats.json'

suncg_sub_dir = os.listdir(g_suncg_house_path)
for house_dir in suncg_sub_dir:
    house_path = os.path.join(g_suncg_house_path,house_dir,'house.json')
    stats_path = os.path.join(g_suncg_relation_path,house_dir,house_dir+'.stats.json')
    os.getcwd()
    os.chdir('%s' %(os.path.join(g_suncg_house_path,house_dir)))
    pwd_cmd = 'pwd'
    os.system('%s' % (pwd_cmd))
    gaps_cmd = '%s %s -input_stats %s -output_hier %s' %(stats2hie_path, house_path, stats_path, os.path.join(g_suncg_house_path,house_dir))
    print(gaps_cmd)
    os.system('%s' % (gaps_cmd))
