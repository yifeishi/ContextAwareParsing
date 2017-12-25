import torch
from torch.utils import data
from scipy.io import loadmat
from enum import Enum
import numpy as np

def FlipBoxOps(boxes,ops):
    # boxes
#    boxes.reverse()
    record = -1
    for i in range(0,len(boxes)):
        if boxes[i].numpy()[0][0] != 0:
            record = i
            break
    print('has %d boxes, '%(len(boxes)-record),end='')
    while 1:
        if len(boxes)-1 <= 0:
            break
        if boxes[len(boxes)-1].numpy()[0][0] != 0:
            break
        if boxes[len(boxes)-1].numpy()[0][0] == 0:
            boxes.pop()
    boxes_new = boxes

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
    print('%d nodes'%(ops.size()[1]-record))

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
    return (boxes_new,ops_new)

class Tree(object):
    class NodeType(Enum):
        BOX = 0  # box node
        ADJ = 1  # adjacency (adjacent part assembly) node
        SYM = 2  # symmetry (symmetric part grouping) node

    class Node(object):
        def __init__(self, box=None, left=None, right=None, node_type=None, sym=None):
            self.box = box          # box feature vector for a leaf node
            self.sym = sym          # symmetry parameter vector for a symmetry node
            self.left = left        # left child for ADJ or SYM (a symmeter generator)
            self.right = right      # right child
            self.node_type = node_type
            self.label = torch.LongTensor([self.node_type.value])

        def is_leaf(self):
            return self.node_type == Tree.NodeType.BOX and self.box is not None

        def is_adj(self):
            return self.node_type == Tree.NodeType.ADJ

        def is_sym(self):
            return self.node_type == Tree.NodeType.SYM


    def __init__(self, boxes, ops):
        box_list = [b for b in torch.split(boxes, 1, 0)]
        (box_list,ops) = FlipBoxOps(box_list,ops)
        out1 = box_list[0].numpy()

        queue = []
        for id in range(ops.size()[1]):
            if ops[0, id] == Tree.NodeType.BOX.value:
#                print('add leaf')
                box=box_list.pop()
                queue.append(Tree.Node(box, node_type=Tree.NodeType.BOX))
            elif ops[0, id] == Tree.NodeType.ADJ.value:
#                print('add internal')
                left_node = queue.pop()
                right_node = queue.pop()
                queue.append(Tree.Node(left=left_node, right=right_node, node_type=Tree.NodeType.ADJ))
            elif ops[0, id] == Tree.NodeType.SYM.value:
                node = queue.pop()
                queue.append(Tree.Node(left=node, sym=sym_param.pop(), node_type=Tree.NodeType.SYM))
            else:
                break
        assert len(queue) == 1
        self.root = queue[0]


class GRASSDataset(data.Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        box_data = torch.from_numpy(loadmat(self.dir+'/box_data.mat')['boxes']).float()
        op_data = torch.from_numpy(loadmat(self.dir+'/op_data.mat')['ops']).int()
        sym_data = torch.from_numpy(loadmat(self.dir+'/sym_data.mat')['syms']).float()

        #weight_list = torch.from_numpy(loadmat(self.dir+'/weights.mat')['weights']).float()
        num_examples = op_data.size()[1]
        box_data = torch.chunk(box_data, num_examples, 1)
        op_data = torch.chunk(op_data, num_examples, 1)
        sym_data = torch.chunk(sym_data, num_examples, 1)
        #weight_list = torch.chunk(weight_list, num_examples, 1)
        self.transform = transform
        self.trees = []
        for i in range(len(op_data)) :
            boxes = torch.t(box_data[i])
            ops = torch.t(op_data[i])

            tree = Tree(boxes, ops)
            self.trees.append(tree)
            print('xxxxxxxxxxxxxxxxxxxxxxx one training data')
            break
    def __getitem__(self, index):
        tree = self.trees[index]
        return tree

    def __len__(self):
        return len(self.trees)