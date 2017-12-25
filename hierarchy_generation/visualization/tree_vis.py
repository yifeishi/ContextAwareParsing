# Import Tree instance and faces module
from ete3 import Tree, faces, TreeStyle
import os

g_suncg_house_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/house'
g_suncg_object_path = '/n/fs/deepfusion/users/yifeis/sceneparsing/data/object'

def tree_vis(hier_name):
    # build an tree
    t = Tree()
    file_object1 = open(hier_name,'r')
    try:
        count = 0
        while True:
            line = file_object1.readline()
            if line:
                L = line.split()
                if L[2] != "null":
                    L[1] = L[1]+" "+L[2]
                if count < 2:
                    L[1] = t.add_child(name=L[1],dist=1000,support=1000)
                else:
                    node = t.search_nodes(name=L[0])[0]
                    L[1] = node.add_child(name=L[1],dist=1000,support=1000) 
                count = count + 1
            else:
                break
    finally:
        file_object1.close()
    return t 
    print(t)

nameFace = faces.AttrFace("name", fsize=20, fgcolor="#009000")
def mylayout(node):
    if node.is_leaf():
#        faces.add_face_to_node(nameFace, node, column=0)
        node.img_style["size"] = 80
        node.img_style["shape"] = "sphere"
        node.img_style["fgcolor"] = "#000000"
        L = node.name.split(' ')
        if len(L) > 1:
            img_name = os.path.join(g_suncg_object_path,L[1],"rgb_img","8_rotation.png")
            objectFace = faces.ImgFace(img_name)
            faces.add_face_to_node(objectFace, node, column=0)
    else:
        node.img_style["size"] = 80
        node.img_style["shape"] = "sphere"
        node.img_style["fgcolor"] = "#AA0000"



suncg_sub_dir = os.listdir(g_suncg_house_path)
for house_dir in suncg_sub_dir:
    print(house_dir)
    file_names = os.listdir(os.path.join(g_suncg_house_path,house_dir))
    for file_name in file_names:
        L = file_name.split('_')
        if L[0] != 'hier':
            continue
        img_out_name = file_name.split('.')[0] + '.png'
        t = tree_vis(os.path.join(g_suncg_house_path,house_dir,file_name))
        ts = TreeStyle()
        ts.layout_fn = mylayout
        ts.show_leaf_name = True
        ts.scale =  0.5
        ts.rotation = 90
        t.render(os.path.join(g_suncg_house_path,house_dir,img_out_name), w=2000, tree_style = ts)   

