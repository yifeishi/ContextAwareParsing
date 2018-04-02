import torch
from torch import nn
from torch.autograd import Variable
import util
import scipy.io as sio
from grassdata import GRASSDataset
from grassdata import GRASSDatasetTest
from grassmodel import GRASSEncoderDecoder
from grassdata import GRASSDataset
import grassmodel
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# xMax,yMax,zMax,xMin,yMin,zMin to size,cen
def ObbFeatureTransformer(obj_obb_fea_tmp):
    obj_obb_fea = np.zeros(6)
    obj_obb_fea[0] = obj_obb_fea_tmp[0] - obj_obb_fea_tmp[3]
    obj_obb_fea[1] = obj_obb_fea_tmp[1] - obj_obb_fea_tmp[4]
    obj_obb_fea[2] = obj_obb_fea_tmp[2] - obj_obb_fea_tmp[5]
    obj_obb_fea[3] = (obj_obb_fea_tmp[0] + obj_obb_fea_tmp[3])*0.5
    obj_obb_fea[4] = (obj_obb_fea_tmp[1] + obj_obb_fea_tmp[4])*0.5
    obj_obb_fea[5] = (obj_obb_fea_tmp[2] + obj_obb_fea_tmp[5])*0.5
    return obj_obb_fea

# size,cen to xMax,yMax,zMax,xMin,yMin,zMin
def ObbFeatureTransformerReverse(obj_obb_fea_tmp):
    obj_obb_fea = np.zeros(6)
    obj_obb_fea[0] = obj_obb_fea_tmp[3] + obj_obb_fea_tmp[0]*0.5
    obj_obb_fea[1] = obj_obb_fea_tmp[4] + obj_obb_fea_tmp[1]*0.5
    obj_obb_fea[2] = obj_obb_fea_tmp[5] + obj_obb_fea_tmp[2]*0.5
    obj_obb_fea[3] = obj_obb_fea_tmp[3] - obj_obb_fea_tmp[0]*0.5
    obj_obb_fea[4] = obj_obb_fea_tmp[4] - obj_obb_fea_tmp[1]*0.5
    obj_obb_fea[5] = obj_obb_fea_tmp[5] - obj_obb_fea_tmp[2]*0.5
    return obj_obb_fea

config = util.get_args()
config.cuda = not config.no_cuda
if config.gpu<0 and config.cuda:
    config.gpu = 0
torch.cuda.set_device(config.gpu)
if config.cuda and torch.cuda.is_available():
    print("Using CUDA on GPU ", config.gpu)
else:
    print("Not using CUDA.")

encoder_decoder = torch.load('./models/snapshots_2018-02-04_04-19-42/encoder_decoder_model_epoch_10_loss_0.2326.pkl')
encoder_decoder.cuda()

grass_data = GRASSDatasetTest(config.data_path)
def my_collate(batch):
    return batch
test_iter = torch.utils.data.DataLoader(grass_data, batch_size=1, shuffle=False, collate_fn=my_collate)

accuracy_sum = 0
num_sum = 0
for batch_idx, batch in enumerate(test_iter):
    print('batch_idx: %d'%batch_idx)
    accuracy, iou, num, init_obb, gt_reg, predict_reg = grassmodel.encode_decode_structure(encoder_decoder, batch[0])
    accuracy_sum = accuracy_sum + accuracy
    num_sum = num_sum + num
    print('accuracy///////////')
    print(float(accuracy)/num)
    print(float(accuracy_sum)/num_sum)
#    print('box///////////')
#    print(init_obb)
#    print(gt_reg)
#    print(predict_reg)
#    out_file_mat_path = 'ouput'
#    sio.savemat(out_file_mat_path, mdict={'init_obb': init_obb,'gt_reg': gt_reg,'predict_reg': predict_reg}, do_compression=True)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    xMax=-999999
    xMin=999999
    yMax=-999999
    yMin=999999
    for i in range(0,init_obb.shape[0]):
        if init_obb[i,0] == 1:
            continue
        obj_obb_fea_tmp_init = init_obb[i,:]
        obj_obb_fea_tmp_gt = init_obb[i,:] + gt_reg[i,:]
        obj_obb_fea_tmp_predict = init_obb[i,:] + predict_reg[i,:]
        
        obj_obb_fea_tmp_init[3:6] = obj_obb_fea_tmp_gt[3:6]
        obj_obb_fea_tmp_predict[3:6] = obj_obb_fea_tmp_gt[3:6]

 #       obj_obb_fea_tmp_init[0:3] = obj_obb_fea_tmp_gt[0:3]
 #       obj_obb_fea_tmp_predict[0:3] = obj_obb_fea_tmp_gt[0:3]


        obj_obb_fea_init = ObbFeatureTransformerReverse(obj_obb_fea_tmp_init)
        obj_obb_fea_gt = ObbFeatureTransformerReverse(obj_obb_fea_tmp_gt)
        obj_obb_fea_predict = ObbFeatureTransformerReverse(obj_obb_fea_tmp_predict)

        ax1.add_patch(patches.Rectangle((obj_obb_fea_init[3], obj_obb_fea_init[5]),obj_obb_fea_tmp_init[0],obj_obb_fea_tmp_init[2],linewidth=1, edgecolor='r', facecolor='none'))
        ax1.add_patch(patches.Rectangle((obj_obb_fea_gt[3], obj_obb_fea_gt[5]),obj_obb_fea_tmp_gt[0],obj_obb_fea_tmp_gt[2],linewidth=1, edgecolor='b', facecolor='none'))
        ax1.add_patch(patches.Rectangle((obj_obb_fea_predict[3], obj_obb_fea_predict[5]),obj_obb_fea_tmp_predict[0],obj_obb_fea_tmp_predict[2],linewidth=1, edgecolor='g', facecolor='none'))
        
        if xMin>obj_obb_fea_gt[3]:
            xMin=obj_obb_fea_gt[3]
        if yMin>obj_obb_fea_gt[5]:
            yMin=obj_obb_fea_gt[5]
        if xMax<obj_obb_fea_gt[3]+obj_obb_fea_tmp_gt[0]:
            xMax=obj_obb_fea_gt[3]+obj_obb_fea_tmp_gt[0]
        if yMax<obj_obb_fea_gt[5]+obj_obb_fea_tmp_gt[2]:
            yMax=obj_obb_fea_gt[5]+obj_obb_fea_tmp_gt[2]
        
    xMin=xMin-1
    yMin=yMin-1
    xMax=xMax+1
    yMax=yMax+1
    plt.xlim(xMin,xMax)
    plt.ylim(yMin,yMax)
    #fig1.savefig('./plot/rect'+str(batch_idx)+'.png', dpi=90, bbox_inches='tight')