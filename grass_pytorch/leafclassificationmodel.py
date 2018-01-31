import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time
import numpy as np

class BoxEncoder(nn.Module):
    def __init__(self, input_size, feature_size):
        super(BoxEncoder, self).__init__()
        self.encoder = nn.Linear(input_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, box_input):
        box_vector = self.encoder(box_input[:,0:2054])
        box_vector = self.tanh(box_vector)
        return box_vector # 1000

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


#########################################################################################
## LeafClassification
#########################################################################################
class LeafClassification(nn.Module):
    def __init__(self, config):
        super(LeafClassification, self).__init__()
        self.box_encoder = BoxEncoder(input_size = config.obj_code_size, feature_size = config.feature_size)
        self.node_classifier = NodeClassifier(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.mseLoss = nn.MSELoss()
        self.creLoss = nn.CrossEntropyLoss()

    def boxEncoder(self, box):
        return self.box_encoder(box)

    def nodeClassifier(self, feature):
        return self.node_classifier(feature)

    def classifyLossEstimator(self, label_vector, gt_label_vector):
        return torch.cat([self.creLoss(l.unsqueeze(0), gt).mul(0.2) for l, gt in zip(label_vector, gt_label_vector)], 0)
        
    def vectorAdder(self, v1, v2):
        return v1.add_(v2)

    def vectorMultipler(self, v):
        return v.mul_(0.2)
    
    def tensor2Node(self, v):
        return v

def leaf_classification_fold(fold, tree):
    def node_classification(node):
        if node.is_leaf():
            feature = fold.add('boxEncoder', node.box)
            label = fold.add('nodeClassifier', feature)
            label_loss = fold.add('classifyLossEstimator', label, node.category)
            return label_loss
        elif node.is_adj():
            left_loss = node_classification(node.left)
            right_loss = node_classification(node.right)
            loss = fold.add('vectorAdder', left_loss, right_loss)
            return loss

    loss = node_classification(tree.root)
    return loss

class LeafClassificationTest(nn.Module):
    def __init__(self, config):
        super(LeafClassificationTest, self).__init__()
        self.box_encoder = BoxEncoder(input_size = config.obj_code_size, feature_size = config.feature_size)
        self.node_classifier = NodeClassifier(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.creLoss = nn.CrossEntropyLoss()

    def boxEncoder(self, box):
        return self.box_encoder(box)

    def nodeClassifier(self, feature):
        return self.node_classifier(feature)

    def classifyLossEstimator(self, label_vector, gt_label_vector):
        return torch.cat([self.creLoss(l.unsqueeze(0), gt).mul(0.2) for l, gt in zip(label_vector, gt_label_vector)], 0)

    def vectorAdder(self, v1, v2):
        return v1.add_(v2)

    def vectorMultipler(self, v):
        return v.mul_(0.2)
    
    def tensor2Node(self, v):
        return v

def leaf_classification_structure(model, tree):
    def node_classification(node):
        if node.is_leaf():
            node.box = Variable(node.box.cuda(), requires_grad=False)
            feature = model.boxEncoder(node.box)
            label_feature = model.nodeClassifier(feature)
            # prediction
            value, label = torch.max(label_feature, 1)
#            print('..............label')
#            print(label.data.cpu().numpy()[0])
#            print(node.category.numpy()[0])
            if label.data.cpu().numpy()[0] == node.category.numpy()[0]:
                accuracy = 1
            else:
                accuracy = 0
            num = 1
            return accuracy, num
        elif node.is_adj():
            left_accuracy, left_num = node_classification(node.left)
            right_accuracy, right_num = node_classification(node.right)
            accuracy = left_accuracy + right_accuracy
            num = left_num + right_num
            return accuracy, num

    accuracy, num = node_classification(tree.root)
    return accuracy, num