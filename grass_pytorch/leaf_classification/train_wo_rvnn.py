import time
import os
from time import gmtime, strftime
from datetime import datetime
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
from torchfoldext import FoldExt
import util
from dynamicplot import DynamicPlot
from grassdata import GRASSDataset
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


leaf_classification = LeafClassification(config)
#leaf_classification = torch.load('./models/snapshots_2018-01-05_20-36-40/encoder_decoder_model_epoch_50_loss_0.1429.pkl')


if config.cuda:
    leaf_classification.cuda()

print("Loading data ...... \n", end='', flush=True)
grass_data = GRASSDataset(config.data_path)
def my_collate(batch):
    return batch
train_iter = torch.utils.data.DataLoader(grass_data, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate)
print("DONE")


print("Start training ...... ")
start = time.time()

if config.save_snapshot:
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    snapshot_folder = os.path.join(config.save_path, 'snapshots_'+strftime("%Y-%m-%d_%H-%M-%S",gmtime()))
    if not os.path.exists(snapshot_folder):
        os.makedirs(snapshot_folder)

if config.save_log:
    fd_log = open('training_log.log', mode='a')
    fd_log.write('\n\nTraining log at '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    fd_log.write('\n#epoch: {}'.format(config.epochs))
    fd_log.write('\nbatch_size: {}'.format(config.batch_size))
    fd_log.write('\ncuda: {}'.format(config.cuda))
    fd_log.flush()

header = '     Time    Epoch     Iteration    Progress(%)    LR       LabelLoss  TotalLoss'
log_template = ' '.join('{:>9s},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>9.1f}%,{:>11.9f},{:>10.4f},{:>10.4f}'.split(','))
print(header)
total_iter = config.epochs * len(train_iter)


learning_rate = config.lr
count = 0
for epoch in range(config.epochs):
    encoder_decoder_opt = torch.optim.Adam(leaf_classification.parameters(), lr=learning_rate)
    for batch_idx, batch in enumerate(train_iter):
        enc_fold = FoldExt(cuda=config.cuda)
        enc_dec_fold_nodes = []
        for example in batch:
            enc_dec_fold_nodes.append(leafclassificationmodel.leaf_classification_fold(enc_fold, example))
        total_loss = enc_fold.apply(leaf_classification, [enc_dec_fold_nodes])
       
        label_loss = total_loss[0].sum() / len(batch)
        
        encoder_decoder_opt.zero_grad()
        label_loss.backward()
        encoder_decoder_opt.step()

        # Report statistics
        if batch_idx % config.show_log_every == 0:
            print(log_template.format(strftime("%H:%M:%S",time.gmtime(time.time()-start)),
                epoch, config.epochs, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx+len(train_iter)*epoch) / (len(train_iter)*config.epochs),
                learning_rate, label_loss.data[0], label_loss.data[0]))
        count=count+1
        if count%5000==4999:
            learning_rate = learning_rate * 0.5

    # Save snapshots of the models being trained
    if config.validate and (epoch+1) % config.validate_every == 0 :
        print("Test on validation set ...... ", end='', flush=True)
        
    # Save snapshots of the models being trained
    if config.save_snapshot and (epoch+0) % config.save_snapshot_every == 0 :
        print("Saving snapshots of the models ...... ", end='', flush=True)
        torch.save(leaf_classification, snapshot_folder+'/encoder_decoder_model_epoch_{}_loss_{:.4f}.pkl'.format(epoch+0, label_loss.data[0]))
        print("DONE")
   
# Save the final models
print("Saving final models ...... ", end='', flush=True)
torch.save(leaf_classification, config.save_path+'/encoder_decoder_model.pkl')
print("DONE")