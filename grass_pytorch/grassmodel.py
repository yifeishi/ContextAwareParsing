import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time
import numpy as np

#########################################################################################
## Encoder
#########################################################################################

def layout(box1, box2):
    layout_code = torch.Tensor(box1.size()[0],7).zero_()
    merge_box_code = torch.Tensor(box1.size()[0],6).zero_()
    
    box1 = box1.data.cpu()
    box2 = box2.data.cpu()

    layout_code[:,0] = box1[:,5] - box2[:,5]
    layout_code[:,1] = box1[:,5] - box2[:,2]
    layout_code[:,2] = box1[:,2] - box2[:,5]
    layout_code[:,3] = ((box1[:,0]+box1[:,3])*0.5-(box2[:,0]+box2[:,3])*0.5)*((box1[:,0]+box1[:,3])*0.5-(box2[:,0]+box2[:,3])*0.5)\
                + ((box1[:,1]+box1[:,4])*0.5-(box2[:,1]+box2[:,4])*0.5)*((box1[:,1]+box1[:,4])*0.5-(box2[:,1]+box2[:,4])*0.5)\
                + ((box1[:,2]+box1[:,5])*0.5-(box2[:,2]+box2[:,5])*0.5)*((box1[:,2]+box1[:,5])*0.5-(box2[:,2]+box2[:,5])*0.5)
    layout_code[:,3] = layout_code[:,3]**0.5
    
    for j in range(0,3):
        for i in range(0,box1.size()[0]):
            if box1[i,j] > box2[i,j]:
                merge_box_code[i,j] = box1[i,j]
            else:
                merge_box_code[i,j] = box2[i,j]
    for j in range(3,6):
        for i in range(0,box1.size()[0]):
            if box1[i,j] < box2[i,j]:
                merge_box_code[i,j] = box1[i,j]
            else:
                merge_box_code[i,j] = box2[i,j]
    
    layout_code = Variable(layout_code.cuda(), requires_grad=True)
    merge_box_code = Variable(merge_box_code.cuda(), requires_grad=True)
    return (layout_code, merge_box_code)

class Sampler(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(Sampler, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, feature_size)
        self.mlp2var = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        output = input[:,0:1000]
        return output

class BoxEncoder(nn.Module):
    def __init__(self, input_size, feature_size):
        super(BoxEncoder, self).__init__()
        self.encoder = nn.Linear(input_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, box_input):
        box_vector = self.encoder(box_input[:,0:2054])
        box_vector = self.tanh(box_vector)
        box_vector = torch.cat((box_vector, box_input[:,2054:2060]), 1)
        return box_vector # 1006

class AdjEncoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(AdjEncoder, self).__init__()
        self.left = nn.Linear(feature_size, hidden_size)
        self.right = nn.Linear(feature_size, hidden_size, bias=False)
        self.second = nn.Linear(hidden_size+100, feature_size)
        self.tanh = nn.Tanh()
        self.layoutFC = nn.Linear(7, 100)

    def forward(self, left_input, right_input):
        output = self.left(left_input[:,0:1000])
        output += self.right(right_input[:,0:1000])
        output = self.tanh(output)
        (layout_code, merge_box_code) = layout(left_input[:,1000:1006],right_input[:,1000:1006]) # define
        layout_feature = self.layoutFC(layout_code)
        output = torch.cat((output,layout_feature),1)
        output = self.second(output) # define
        output = self.tanh(output)
        output = torch.cat((output, merge_box_code), 1)
        return output

class SymEncoder(nn.Module):
    def __init__(self, feature_size, symmetry_size, hidden_size):
        super(SymEncoder, self).__init__()
        self.left = nn.Linear(feature_size, hidden_size)
        self.right = nn.Linear(symmetry_size, hidden_size)
        self.second = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, left_input, right_input):
        output = self.left(left_input)
        output += self.right(right_input)
        output = self.tanh(output)
        output = self.second(output)
        output = self.tanh(output)
        return output


#########################################################################################
## Decoder
#########################################################################################

class NodeClassifier(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(NodeClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.tanh = nn.Tanh()
        self.mlp2 = nn.Linear(hidden_size, 100)
        self.softmax = nn.Softmax()

    def forward(self, input_feature):
        output = self.mlp1(input_feature)
        output = self.tanh(output)
        output = self.mlp2(output)

        return output

class SampleDecoder(nn.Module):
    """ Decode a randomly sampled noise into a feature vector """
    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()
        
    def forward(self, input_feature):
        output = self.tanh(self.mlp1(input_feature))
        output = self.tanh(self.mlp2(output))
        return output

class AdjDecoder(nn.Module):
    """ Decode an input (parent) feature into a left-child and a right-child feature """
    def __init__(self, feature_size, hidden_size):
        super(AdjDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, hidden_size)
        self.mlp_left_encode = nn.Linear(feature_size+6, hidden_size)
        self.mlp_left = nn.Linear(hidden_size, feature_size)
        self.mlp_right = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, parent_feature, left_encode_feature, right_encode_feature): # def forward(self, parent_feature, left_encode_feature, right_encode_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        vector_left = self.mlp_left_encode(left_encode_feature[:,0:1006])
        vector_left = self.tanh(vector_left)
        vector_right = self.mlp_left_encode(right_encode_feature[:,0:1006])
        vector_right = self.tanh(vector_right)
        left_feature = self.mlp_left(vector) + self.mlp_left(vector_left)
        left_feature = self.tanh(left_feature)
        right_feature = self.mlp_right(vector)  + self.mlp_right(vector_right)
        right_feature = self.tanh(right_feature)
        return left_feature, right_feature

class SymDecoder(nn.Module):

    def __init__(self, feature_size, symmetry_size, hidden_size):
        super(SymDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, hidden_size) # layer for decoding a feature vector 
        self.tanh = nn.Tanh()
        self.mlp_sg = nn.Linear(hidden_size, feature_size) # layer for outputing the feature of symmetry generator
        self.mlp_sp = nn.Linear(hidden_size, symmetry_size) # layer for outputing the vector of symmetry parameter

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        sym_gen_vector = self.mlp_sg(vector)
        sym_gen_vector = self.tanh(sym_gen_vector)
        sym_param_vector = self.mlp_sp(vector)
        sym_param_vector = self.tanh(sym_param_vector)
        return sym_gen_vector, sym_param_vector

class BoxDecoder(nn.Module):
    def __init__(self, feature_size, box_size):
        super(BoxDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, box_size)
        self.tanh = nn.Tanh()

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        return vector


#########################################################################################
## Functions for model testing: Decode a root code into a tree structure of boxes
#########################################################################################

def vrrotvec2mat(rotvector):
    s = math.sin(rotvector[3])
    c = math.cos(rotvector[3])
    t = 1 - c
    x = rotvector[0]
    y = rotvector[1]
    z = rotvector[2]
    m = torch.FloatTensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]]).cuda()
    return m

def decode_structure(model, root_code):
    """
    Decode a root code into a tree structure of boxes
    """
    decode = model.sampleDecoder(root_code)
    syms = [torch.ones(8).mul(10).cuda()]
    stack = [decode]
    boxes = []
    while len(stack) > 0:
        f = stack.pop()
        label_prob = model.nodeClassifier(f)
        _, label = torch.max(label_prob, 1)
        label = label.data
        if label[0] == 1:  # ADJ
            left, right = model.adjDecoder(f)
            stack.append(left)
            stack.append(right)
            s = syms.pop()
            syms.append(s)
            syms.append(s)
        if label[0] == 2:  # SYM
            left, s = model.symDecoder(f)
            s = s.squeeze(0)
            stack.append(left)
            syms.pop()
            syms.append(s.data)
        if label[0] == 0:  # BOX
            reBox = model.boxDecoder(f)
            reBoxes = [reBox]
            s = syms.pop()
            l1 = abs(s[0] + 1)
            l2 = abs(s[0])
            l3 = abs(s[0] - 1)

            if l1 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.data.squeeze(0), 1, 0)
                f1 = torch.cat([sList[1], sList[2], sList[3]])
                f1 = f1/torch.norm(f1)
                f2 = torch.cat([sList[4], sList[5], sList[6]])
                folds = round(1/s[7])
                for i in range(folds-1):
                    rotvector = torch.cat([f1, sList[7].mul(2*3.1415).mul(i+1)])
                    rotm = vrrotvec2mat(rotvector)
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    dir0 = torch.cat([bList[3], bList[4], bList[5]])
                    dir1 = torch.cat([bList[6], bList[7], bList[8]])
                    dir2 = torch.cat([bList[9], bList[10], bList[11]])
                    newcenter = rotm.matmul(center.add(-f2)).add(f2)
                    newdir1 = rotm.matmul(dir1)
                    newdir2 = rotm.matmul(dir2)
                    newbox = torch.cat([newcenter, dir0, newdir1, newdir2])
                    reBoxes.append(Variable(newbox.unsqueeze(0)))

            if l2 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.data.squeeze(0), 1, 0)
                trans = torch.cat([sList[1], sList[2], sList[3]])
                trans_end = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                trans_length = math.sqrt(torch.sum(trans**2))
                trans_total = math.sqrt(torch.sum(trans_end.add(-center)**2))
                folds = round(trans_total/trans_length)
                for i in range(folds):
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    dir0 = torch.cat([bList[3], bList[4], bList[5]])
                    dir1 = torch.cat([bList[6], bList[7], bList[8]])
                    dir2 = torch.cat([bList[9], bList[10], bList[11]])
                    newcenter = center.add(trans.mul(i+1))
                    newbox = torch.cat([newcenter, dir0, dir1, dir2])
                    reBoxes.append(Variable(newbox.unsqueeze(0)))

            if l3 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.data.squeeze(0), 1, 0)
                ref_normal = torch.cat([sList[1], sList[2], sList[3]])
                ref_normal = ref_normal/torch.norm(ref_normal)
                ref_point = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                dir0 = torch.cat([bList[3], bList[4], bList[5]])
                dir1 = torch.cat([bList[6], bList[7], bList[8]])
                dir2 = torch.cat([bList[9], bList[10], bList[11]])
                if ref_normal.matmul(ref_point.add(-center)) < 0:
                    ref_normal = -ref_normal
                newcenter = ref_normal.mul(2*abs(torch.sum(ref_point.add(-center).mul(ref_normal)))).add(center)
                if ref_normal.matmul(dir1) < 0:
                    ref_normal = -ref_normal
                dir1 = dir1.add(ref_normal.mul(-2*ref_normal.matmul(dir1)))
                if ref_normal.matmul(dir2) < 0:
                    ref_normal = -ref_normal
                dir2 = dir2.add(ref_normal.mul(-2*ref_normal.matmul(dir2)))
                newbox = torch.cat([newcenter, dir0, dir1, dir2])
                reBoxes.append(Variable(newbox.unsqueeze(0)))

            boxes.extend(reBoxes)

    return boxes

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2', 'z1', 'z2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2', 'z1', 'z2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb1['z1'] < bb1['z2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    assert bb2['z1'] < bb2['z2']
    
    # determine the coordinates of the intersection rectangle
    x_min = max(bb1['x1'], bb2['x1'])
    y_min = max(bb1['y1'], bb2['y1'])
    z_min = max(bb1['z1'], bb2['z1'])
    x_max = min(bb1['x2'], bb2['x2'])
    y_max = min(bb1['y2'], bb2['y2'])
    z_max = min(bb1['z2'], bb2['z2'])

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    
    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1']) * (bb1['z2'] - bb1['z1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1']) * (bb2['z2'] - bb2['z1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    """
    print('..............intersection_area')
    print(bb1_area)
    print(bb2_area)
    print(intersection_area)
    """
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

#########################################################################################
## GRASSEncoderDecoder
#########################################################################################
class GRASSEncoderDecoder(nn.Module):
    def __init__(self, config):
        super(GRASSEncoderDecoder, self).__init__()
        self.box_encoder = BoxEncoder(input_size = config.obj_code_size, feature_size = config.feature_size)
        self.adj_encoder = AdjEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.sym_encoder = SymEncoder(feature_size = config.feature_size, symmetry_size = config.symmetry_size, hidden_size = config.hidden_size)
        self.sample_encoder = Sampler(feature_size = config.feature_size+6, hidden_size = config.hidden_size)

        self.box_decoder = BoxDecoder(feature_size = config.feature_size, box_size = config.box_code_size)
        self.adj_decoder = AdjDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.sym_decoder = SymDecoder(feature_size = config.feature_size, symmetry_size = config.symmetry_size, hidden_size = config.hidden_size)
        self.sample_decoder = SampleDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.node_classifier = NodeClassifier(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.mseLoss = nn.L1Loss(size_average=True)
        self.creLoss = nn.CrossEntropyLoss()

    def boxEncoder(self, box):
        return self.box_encoder(box)

    def adjEncoder(self, left, right):
        return self.adj_encoder(left, right)

    def symEncoder(self, feature, sym):
        return self.sym_encoder(feature, sym)

    def sampleEncoder(self, feature):
        return self.sample_encoder(feature)

    def boxDecoder(self, feature):
        return self.box_decoder(feature)

    def adjDecoder(self, feature, feature_left, feature_right):
        return self.adj_decoder(feature, feature_left, feature_right)

    def symDecoder(self, feature):
        return self.sym_decoder(feature)

    def sampleDecoder(self, feature):
        return self.sample_decoder(feature)

    def nodeClassifier(self, feature):
        return self.node_classifier(feature)

    def boxLossEstimator(self, box_feature, gt_box_reg, gt_box_feature):
        gt_box_feature_last6 = np.zeros(box_feature.shape)
        for i in range(0,6):
            for j in range(0,gt_box_feature_last6.shape[0]):
                gt_box_feature_last6[j][i] = gt_box_feature[j][2048+i].data.cpu().numpy()
        gt_box_feature_last6 = Variable(torch.from_numpy(gt_box_feature_last6).float().cuda(), requires_grad=False)
        for i in range(0,gt_box_feature_last6.shape[0]):
            if gt_box_feature_last6[i][0].data.cpu().numpy() == 1:
                box_feature[i] = gt_box_reg[i]

        
#        print('............box')
#        print(box_feature)
#        print('gt_box_reg')
#        print(gt_box_reg)
#        print('gt_box_feature_last6')
#        print(gt_box_feature_last6)
        
#        return torch.cat([self.mseLoss(b, gt).mul(0.4) for b, gt in zip(box_feature, gt_box_reg)], 0)
        return torch.cat([self.mseLoss(b, gt) for b, gt in zip(box_feature, gt_box_reg)], 0)

    def boxIOUEstimator(self, box_feature, gt_box_feature):
 #       print('...............boxIOUEstimator')
        gt_box_feature_last6 = np.zeros(box_feature.shape)
        for i in range(0,6):
            for j in range(0,gt_box_feature_last6.shape[0]):
                gt_box_feature_last6[j][i] = gt_box_feature[j][2048+i].data.cpu().numpy()
        gt_box_feature_last6 = Variable(torch.from_numpy(gt_box_feature_last6).float().cuda(), requires_grad=False)
        if gt_box_feature_last6[0][0].data.cpu().numpy() == 1:
            box_feature = gt_box_feature_last6
        
        box = np.zeros(6)
        box[0] = box_feature[0,3] + box_feature[0,0]*0.5
        box[1] = box_feature[0,4] + box_feature[0,1]*0.5
        box[2] = box_feature[0,5] + box_feature[0,2]*0.5
        box[3] = box_feature[0,3] - box_feature[0,0]*0.5
        box[4] = box_feature[0,4] - box_feature[0,1]*0.5
        box[5] = box_feature[0,5] - box_feature[0,2]*0.5
        box_dict = {'x1': box[3], 'y1': box[4], 'z1': box[5], 'x2': box[0], 'y2': box[1], 'z2': box[2]}
        
        box_gt = np.zeros(6)
        box_gt[0] = gt_box_feature_last6[0,3] + gt_box_feature_last6[0,0]*0.5
        box_gt[1] = gt_box_feature_last6[0,4] + gt_box_feature_last6[0,1]*0.5
        box_gt[2] = gt_box_feature_last6[0,5] + gt_box_feature_last6[0,2]*0.5
        box_gt[3] = gt_box_feature_last6[0,3] - gt_box_feature_last6[0,0]*0.5
        box_gt[4] = gt_box_feature_last6[0,4] - gt_box_feature_last6[0,1]*0.5
        box_gt[5] = gt_box_feature_last6[0,5] - gt_box_feature_last6[0,2]*0.5
        box_gt_dict = {'x1': box_gt[3], 'y1': box_gt[4], 'z1': box_gt[5], 'x2': box_gt[0], 'y2': box_gt[1], 'z2': box_gt[2]}
        """
        print('.............box1')
        print(box)
        print('.............box_gt')
        print(box_gt)
        """
        iou = get_iou(box_dict, box_gt_dict)
        return iou

    def symLossEstimator(self, sym_param, gt_sym_param):
        return torch.cat([self.mseLoss(s, gt).mul(0.5) for s, gt in zip(sym_param, gt_sym_param)], 0)

    def classifyLossEstimator(self, label_vector, gt_label_vector):
        return torch.cat([self.creLoss(l.unsqueeze(0), gt).mul(0.2) for l, gt in zip(label_vector, gt_label_vector)], 0)
        
    def vectorAdder(self, v1, v2):
        return v1.add_(v2)

    def vectorMultipler(self, v):
        return v.mul_(1)
    
    def tensor2Node(self, v):
        return v

def encode_decode_structure_fold(fold, tree):
    def encode_node(node):
        if node.is_leaf():
            return fold.add('boxEncoder', node.box)
        elif node.is_adj():
            left = encode_node(node.left)
            right = encode_node(node.right)
            return fold.add('adjEncoder',left,right)
    def sample_encoder(feature):
        return fold.add('sampleEncoder', feature)

    def decode_node_box(node, feature):
        if node.is_leaf():
            box = fold.add('boxDecoder', feature)
            recon_loss = fold.add('boxLossEstimator', box, node.reg, node.box)
            label = fold.add('nodeClassifier', feature)
            label_loss = fold.add('classifyLossEstimator', label, node.category)
            recon_loss = fold.add('vectorMultipler', recon_loss)
            loss = fold.add('vectorAdder', recon_loss, label_loss)
            return loss
        elif node.is_adj():
            # encode
#            left_encode = encode_node(node.left)
#            right_encode = encode_node(node.right)
            left_encode = encode_node(node.left)
            right_encode = encode_node(node.right)

            left, right = fold.add('adjDecoder', feature, left_encode, right_encode).split(2)
            left_loss = decode_node_box(node.left, left)
            right_loss = decode_node_box(node.right, right)
            loss = fold.add('vectorAdder', left_loss, right_loss)
            return loss
    def sample_decoder(feature):
        return fold.add('sampleDecoder', feature)

    feature1 = encode_node(tree.root)
    feature2 = sample_encoder(feature1)
    feature3 = sample_decoder(feature2)
    loss = decode_node_box(tree.root, feature3)
    return loss

def encode_decode_recon_structure_fold(fold, tree):
    def encode_node(node):
        if node.is_leaf():
            return fold.add('boxEncoder', node.box)
        elif node.is_adj():
            left = encode_node(node.left)
            right = encode_node(node.right)
            return fold.add('adjEncoder',left,right)
    def sample_encoder(feature):
        return fold.add('sampleEncoder', feature)

    def decode_node_box(node, feature):
        if node.is_leaf():
            box = fold.add('boxDecoder', feature)
            recon_loss = fold.add('boxLossEstimator', box, node.reg, node.box)
            label = fold.add('nodeClassifier', feature)
            label_loss = fold.add('classifyLossEstimator', label, node.category)
            recon_loss = fold.add('vectorMultipler', recon_loss)
            loss = fold.add('vectorAdder', recon_loss, label_loss)
            return recon_loss
        elif node.is_adj():
            # encode
            left_encode = encode_node(node.left)
            right_encode = encode_node(node.right)
            left, right = fold.add('adjDecoder', feature, left_encode, right_encode).split(2)
            left_loss = decode_node_box(node.left, left)
            right_loss = decode_node_box(node.right, right)
            label = fold.add('nodeClassifier', feature)
            loss = fold.add('vectorAdder', left_loss, right_loss)
            return loss
    def sample_decoder(feature):
        return fold.add('sampleDecoder', feature)

    feature1 = encode_node(tree.root)
    feature2 = sample_encoder(feature1)
    feature3 = sample_decoder(feature2)
    loss = decode_node_box(tree.root, feature3)
    return loss

def encode_decode_label_structure_fold(fold, tree):
    def encode_node(node):
        if node.is_leaf():
            return fold.add('boxEncoder', node.box)
        elif node.is_adj():
            left = encode_node(node.left)
            right = encode_node(node.right)
            return fold.add('adjEncoder',left,right)
    def sample_encoder(feature):
        return fold.add('sampleEncoder', feature)

    def decode_node_box(node, feature):
        if node.is_leaf():
            box = fold.add('boxDecoder', feature)
            recon_loss = fold.add('boxLossEstimator', box, node.reg, node.box)
            label = fold.add('nodeClassifier', feature)
            label_loss = fold.add('classifyLossEstimator', label, node.category)
            recon_loss = fold.add('vectorMultipler', recon_loss)
            loss = fold.add('vectorAdder', recon_loss, label_loss)
            return label_loss
        elif node.is_adj():
            # encode
            left_encode = encode_node(node.left)
            right_encode = encode_node(node.right)

            left, right = fold.add('adjDecoder', feature, left_encode, right_encode).split(2)
            left_loss = decode_node_box(node.left, left)
            right_loss = decode_node_box(node.right, right)
            label = fold.add('nodeClassifier', feature)
            loss = fold.add('vectorAdder', left_loss, right_loss)
            return loss
    def sample_decoder(feature):
        return fold.add('sampleDecoder', feature)

    feature1 = encode_node(tree.root)
    feature2 = sample_encoder(feature1)
    feature3 = sample_decoder(feature2)
    loss = decode_node_box(tree.root, feature3)
    return loss


#########################################################################################
## GRASSEncoderDecoder Testing
#########################################################################################
def encode_decode_structure(model, tree):
    def encode_node(node):
        if node.is_leaf():
            node.box = Variable(node.box.cuda(), requires_grad=False)
            output = model.boxEncoder(node.box)
            return output
        elif node.is_adj():
            left = encode_node(node.left)
            right = encode_node(node.right)
            output = model.adjEncoder(left,right)
            return output
    def encode_node_variable(node):
        if node.is_leaf():
            output = model.boxEncoder(node.box)
            return output
        elif node.is_adj():
            left = encode_node_variable(node.left)
            right = encode_node_variable(node.right)
            output = model.adjEncoder(left,right)
            return output
    def sample_encoder(feature):
        output = model.sampleEncoder(feature)
        return output

    def decode_node_box(node, feature):
        if node.is_leaf():
            box = model.boxDecoder(feature)
            iou = 1
            """
            print('..................box')
            print(type(box.data.cpu().numpy()))
            print(box.data.cpu().numpy())
            print('..................reg')
            print(type(node.reg.cpu().numpy()))
            print(node.reg.cpu().numpy())
            print('..................box')
            print(type(node.box.data.cpu().numpy()[:,2048:2054]))
            print(node.box.data.cpu().numpy()[:,2048:2054])
            """
#            iou = model.boxIOUEstimator(box, node.box)
#            print('..............iou')
#            print(iou)
            label_feature = model.nodeClassifier(feature)
            value, label = torch.max(label_feature, 1)
#            print('..............label')
#            print(label.data.cpu().numpy()[0])
#            print(node.category.numpy()[0])
            if label.data.cpu().numpy()[0] == node.category.numpy()[0]:
                accuracy = 1
            else:
                accuracy = 0
            num = 1
            return (accuracy, iou, num, node.box.data.cpu().numpy()[:,2048:2054], node.reg.cpu().numpy(), box.data.cpu().numpy())
        elif node.is_adj():
            # encode
            left_encode = encode_node_variable(node.left)
            right_encode = encode_node_variable(node.right)
            
            left_right = model.adjDecoder(feature, left_encode, right_encode)
            left = left_right[0]
            right = left_right[1]
            left_accuracy, left_iou, left_num, left_init_obb, left_gt_reg, left_predict_reg  = decode_node_box(node.left, left)
            right_accuracy, right_iou, right_num, right_init_obb, right_gt_reg, right_predict_reg = decode_node_box(node.right, right)
            accuracy = left_accuracy + right_accuracy
            left_iou= left_iou + right_iou
            num = left_num + right_num
            init_obb = np.vstack((left_init_obb,right_init_obb))
            gt_reg = np.vstack((left_gt_reg,right_gt_reg))
            predict_reg = np.vstack((left_predict_reg,right_predict_reg))
            return (accuracy, left_iou, num, init_obb, gt_reg, predict_reg)
    def sample_decoder(feature):
        output = model.sampleDecoder(feature)
        return output

    feature1 = encode_node(tree.root)
    feature2 = sample_encoder(feature1)
    feature3 = sample_decoder(feature2)
    accuracy, iou, num, init_obb, gt_reg, predict_reg = decode_node_box(tree.root, feature3)
    return accuracy, iou, num, init_obb, gt_reg, predict_reg
    