import torch
from torch.utils import data
from scipy.io import loadmat
from enum import Enum
import numpy as np

def FlipBoxOps(boxes,boxes_reg,ops,categories):
    # boxes
    record = -1
    for i in range(0,len(boxes)):
        if boxes[i].numpy()[0][0] != 0:
            record = i
            break
#    print('has %d boxes, '%(len(boxes)-record),end='')
    while 1:
        if len(boxes)-1 <= 0:
            break
        if boxes[len(boxes)-1].numpy()[0][0] != 0:
            break
        if boxes[len(boxes)-1].numpy()[0][0] == 0:
            boxes.pop()
            boxes_reg.pop()
    
    boxes_new = boxes
    boxes_reg_new = boxes_reg
#    print('has %d boxes, '%len(boxes_new),end='')

    """
    # old code
    boxes_new = boxes
    for i in range(0,record):
        tmp = torch.zeros(boxes[i].size())
        boxes_new[i] = tmp
    count = 0
    for i in range(record,len(boxes)):
        boxes_new[count] = boxes[i]
        count = count + 1
    for i in range(record,len(boxes)):
        tmp = torch.zeros(boxes[i].size())
        boxes_new[i] = tmp
    """
    # ops
    opnumpy = ops.numpy()
    ops = torch.from_numpy(np.fliplr(opnumpy).copy())
    record = -1
    for i in range(0,ops.size()[1]):
        if ops.numpy()[0][i] != -1:           
            record = i
            break
#   print('%d nodes'%(ops.size()[1]-record))

    """
    # old code
    while 1:
        if ops.size()[1]-1 <= 0:
            break
        if ops.numpy()[0][ops.size()[1]-1] != -1:
            break
        if ops.numpy()[0][ops.size()[1]-1] == -1:
            ops.pop()
    print('new ops................')
    print(ops) 
    ops_new = ops
    """

    ops_new = torch.IntTensor(1,ops.size()[1])
    for i in range(0,ops.size()[1]):
        ops_new.numpy()[0][i] = -1
    count = 0
    for i in range(record,ops.size()[1]):
        ops_new.numpy()[0][count] = ops.numpy()[0][i]
        count = count + 1

    """
    # old code
    for i in range(record,ops.size()[1]):
        ops_new.numpy()[0][i] = -1
    """

    # categories
    categoriesnumpy = categories.numpy()
    categories = torch.from_numpy(np.fliplr(categoriesnumpy).copy())
    record = -1
    for i in range(0,categories.size()[1]):
        if categories.numpy()[0][i] != -1:           
            record = i
            break
#    print('%d nodes'%(categories.size()[1]-record))

    categories_new = torch.IntTensor(1,categories.size()[1])
    for i in range(0,categories.size()[1]):
        categories_new.numpy()[0][i] = -1
    count = 0
    for i in range(record,categories.size()[1]):
        categories_new.numpy()[0][count] = categories.numpy()[0][i]
        count = count + 1
    return (boxes_new,boxes_reg_new,ops_new,categories_new)

class Tree(object):
    class NodeType(Enum):
        BOX = 0  # box node
        ADJ = 1  # adjacency (adjacent part assembly) node
        SYM = 2  # symmetry (symmetric part grouping) node

    class Node(object):
        def __init__(self, box=None, left=None, right=None, node_type=None, sym=None, index=None, category=None, feature=None, reg=None):
            
            self.box = box          # box feature vector for a leaf node
            self.sym = sym          # symmetry parameter vector for a symmetry node
            self.left = left        # left child for ADJ or SYM (a symmeter generator)
            self.right = right      # right child
            self.node_type = node_type
            self.label = torch.LongTensor([self.node_type.value])
            self.index = index
            if(category!=None):
                self.category = torch.LongTensor([category])
            self.feature = feature
            self.reg = reg

        def is_leaf(self):
            return self.node_type == Tree.NodeType.BOX and self.box is not None

        def is_adj(self):
            return self.node_type == Tree.NodeType.ADJ

        def is_sym(self):
            return self.node_type == Tree.NodeType.SYM


    def __init__(self, boxes, boxes_reg, ops, categories, index):
        
        box_list = [b for b in torch.split(boxes, 1, 0)]
        box_reg_list = [b for b in torch.split(boxes_reg, 1, 0)]
        (box_list,box_reg_list,ops,categories) = FlipBoxOps(box_list,box_reg_list,ops,categories)#
        
        queue = []
        """
        bug_killer = open("tree"+str(index)+".txt", "w")
        bug_killer.write("init tree...................\n")
        """
        for id in range(ops.size()[1]):
            if ops[0, id] == Tree.NodeType.BOX.value:
                box=box_list.pop()
                box_reg=box_reg_list.pop()
                # set mv feature entry to 0
#                for i in range(0,2048):
#                    box[0].numpy()[i] = 0
#                print('category: %s'%categories[0,id])
#                print('box: %s'%box[0].numpy()[0])
                if box[0].numpy()[0] == 1.0 and categories[0,id] == -1:
#                    print('floor')
#                    print(categories[0,id])
                    queue.append(Tree.Node(box, reg=box_reg, node_type=Tree.NodeType.BOX, index=id, category=99)) # for floor
                else:
#                    print('object')
#                    print(categories[0,id])
                    queue.append(Tree.Node(box, reg=box_reg, node_type=Tree.NodeType.BOX, index=id, category=categories[0,id])) # for object
                """
                bug_killer.write("%d add leaf\n" %id)
                bug_killer.write(str(box.numpy()))
                bug_killer.write("\n\n")
                """
            elif ops[0, id] == Tree.NodeType.ADJ.value:
                left_node = queue.pop()
                right_node = queue.pop()
                queue.append(Tree.Node(left=left_node, right=right_node, node_type=Tree.NodeType.ADJ, index=id))
                """
                bug_killer.write("%d add internal, left node is %d, right node is %d\n\n" %(id,left_node.index,right_node.index))
                """
            else:
                break
        """
        bug_killer.close()
        """
        assert len(queue) == 1
        self.root = queue[0]
        self.index = index


class GRASSDataset(data.Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir

        for i in range(1, 80):
            mat_data = loadmat(self.dir+'/bedroom_room_feature_'+str(i)+'.mat')
            if i == 1:
                box_data = mat_data['boxes']
                box_reg_data = mat_data['boxes_reg']
                op_data = mat_data['ops']
                category_data = mat_data['category']
            else:
                box_data = np.hstack((box_data,mat_data['boxes']))
                box_reg_data = np.hstack((box_reg_data,mat_data['boxes_reg']))
                op_data = np.hstack((op_data,mat_data['ops']))
                category_data = np.hstack((category_data,mat_data['category']))
            print(i)
            print(box_data.shape)
            print(box_reg_data.shape)
            print(op_data.shape)
            print(category_data.shape)
            print('//////////////////////')

        box_data = torch.from_numpy(box_data).float()
        box_reg_data = torch.from_numpy(box_reg_data).float()
        op_data = torch.from_numpy(op_data).int()
        category_data = torch.from_numpy(category_data).int()

        print(box_data.shape)
        print(box_reg_data.shape)
        print(op_data.shape)
        print(category_data.shape)
        print('////////...............//////////////')

        num_examples = op_data.size()[1]
        box_data = torch.chunk(box_data, num_examples, 1)
        box_reg_data = torch.chunk(box_reg_data, num_examples, 1)
        op_data = torch.chunk(op_data, num_examples, 1)
        category_data = torch.chunk(category_data, num_examples, 1)
        
        self.transform = transform
        self.trees = []
        count = 0

#        trainNum = 1000
        for i in range(len(op_data)) :
            boxes = torch.t(box_data[i])
            boxes_reg = torch.t(box_reg_data[i])
            ops = torch.t(op_data[i])
            categories = torch.t(category_data[i])
            tree = Tree(boxes, boxes_reg, ops, categories, i)
            self.trees.append(tree)
            count = count + 1
#            if count >= trainNum:
#                break
    def __getitem__(self, index):
        tree = self.trees[index]
        return tree

    def __len__(self):
        return len(self.trees)


class GRASSDatasetTest(data.Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
    
        for i in range(1, 2):
            mat_data = loadmat(self.dir+'/bedroom_room_feature_'+str(i)+'.mat')
            if i == 1:
                box_data = mat_data['boxes']
                box_reg_data = mat_data['boxes_reg']
                op_data = mat_data['ops']
                category_data = mat_data['category']
            else:
                box_data = np.hstack((box_data,mat_data['boxes']))
                box_reg_data = np.hstack((box_data,mat_data['boxes_reg']))
                op_data = np.hstack((op_data,mat_data['ops']))
                category_data = np.hstack((category_data,mat_data['category']))
            print(i)
            print(box_data.shape)
            print(box_reg_data.shape)
            print(op_data.shape)
            print(category_data.shape)
            print('//////////////////////')

        box_data = torch.from_numpy(box_data).float()
        box_reg_data = torch.from_numpy(box_reg_data).float()
        op_data = torch.from_numpy(op_data).int()
        category_data = torch.from_numpy(category_data).int()

        num_examples = op_data.size()[1]
        box_data = torch.chunk(box_data, num_examples, 1)
        box_reg_data = torch.chunk(box_reg_data, num_examples, 1)
        op_data = torch.chunk(op_data, num_examples, 1)
        category_data = torch.chunk(category_data, num_examples, 1)
        
        self.transform = transform
        self.trees = []
        count = 0
        trainNum = 1000
        for i in range(len(op_data)) :
            boxes = torch.t(box_data[i])
            boxes_reg = torch.t(box_reg_data[i])
            ops = torch.t(op_data[i])
            categories = torch.t(category_data[i])
            tree = Tree(boxes, boxes_reg, ops, categories, i)
            self.trees.append(tree)
            count = count + 1
            if count >= trainNum:
                break
    def __getitem__(self, index):
        tree = self.trees[index]
        return tree

    def __len__(self):
        return len(self.trees)