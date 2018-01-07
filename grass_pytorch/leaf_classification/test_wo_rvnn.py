import torch
from torch import nn
from torch.autograd import Variable
import util
from grassdata import GRASSDataset
from grassdata import GRASSDatasetTest
from leafclassificationmodel import LeafClassification
import leafclassificationmodel


config = util.get_args()
config.cuda = not config.no_cuda
if config.gpu<0 and config.cuda:
    config.gpu = 0
torch.cuda.set_device(config.gpu)
if config.cuda and torch.cuda.is_available():
    print("Using CUDA on GPU ", config.gpu)
else:
    print("Not using CUDA.")

leaf_classification = torch.load('./models/snapshots_2018-01-05_23-32-47/encoder_decoder_model_epoch_410_loss_0.0285.pkl')
print(leaf_classification)
print(leaf_classification.box_encoder.encoder)
leaf_classification.cuda()

grass_data = GRASSDatasetTest(config.data_path)
def my_collate(batch):
    return batch
test_iter = torch.utils.data.DataLoader(grass_data, batch_size=1, shuffle=False, collate_fn=my_collate)

accuracy_sum = 0
num_sum = 0
for batch_idx, batch in enumerate(test_iter):
    accuracy, num = leafclassificationmodel.leaf_classification_structure(leaf_classification, batch[0])
    accuracy_sum = accuracy_sum + accuracy
    num_sum = num_sum + num
#    print('accuracy///////////')
    print(float(accuracy)/num)
#    print(float(accuracy_sum)/num_sum)

print('.............')
print(float(accuracy_sum)/num_sum)

"""
for i in range(10):
    root_code = Variable(torch.randn(1,80)).cuda()
    boxes = grassmodel.decode_structure(decoder, root_code)
    showGenshape(torch.cat(boxes,0).data.cpu().numpy())
"""