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
from grassmodel import GRASSEncoderDecoder
import grassmodel
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


#encoder_decoder = GRASSEncoderDecoder(config)
encoder_decoder = torch.load('./models/snapshots_2018-02-03_21-22-32/encoder_decoder_model_epoch_40_loss_0.1960.pkl')
if config.cuda:
    encoder_decoder.cuda()

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

header = '     Time    Epoch     Iteration    Progress(%)       LR      LabelLoss   ReconLoss   TotalLoss'
log_template = ' '.join('{:>9s},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>9.1f}%,   {:>11.9f},{:>11.4f},{:>10.4f},{:>10.4f}'.split(','))
print(header)
total_iter = config.epochs * len(train_iter)

if not config.no_plot:
    plot_x = [x for x in range(total_iter)]
    plot_total_loss = [None for x in range(total_iter)]
    plot_recon_loss = [None for x in range(total_iter)]
    plot_kldiv_loss = [None for x in range(total_iter)]
    dyn_plot = DynamicPlot(title='Training loss over epochs (GRASS)', xdata=plot_x, ydata={'Total_loss':plot_total_loss, 'Reconstruction_loss':plot_recon_loss, 'KL_divergence_loss':plot_kldiv_loss})
    iter_id = 0
    max_loss = 0

learning_rate = config.lr
count = 0
for epoch in range(config.epochs):
#    encoder_decoder_opt = torch.optim.Adam(encoder_decoder.parameters(), lr=learning_rate)
    encoder_decoder_opt = torch.optim.Adam([
            {'params': encoder_decoder.adj_encoder.parameters()},
            {'params': encoder_decoder.sym_encoder.parameters()},
            {'params': encoder_decoder.sample_encoder.parameters()},
            {'params': encoder_decoder.box_decoder.parameters()},
            {'params': encoder_decoder.adj_decoder.parameters()},
            {'params': encoder_decoder.sym_decoder.parameters()},
            {'params': encoder_decoder.sample_decoder.parameters()},
            {'params': encoder_decoder.box_encoder.encoder.parameters(), 'lr': learning_rate*0.01},
            {'params': encoder_decoder.node_classifier.parameters(), 'lr': learning_rate*0.01}
        ], lr=learning_rate)

    for batch_idx, batch in enumerate(train_iter):
        enc_fold = FoldExt(cuda=config.cuda)
        enc_dec_fold_nodes = []
        enc_dec_recon_fold_nodes = []
        enc_dec_label_fold_nodes = []
        for example in batch:
            enc_dec_fold_nodes.append(grassmodel.encode_decode_structure_fold(enc_fold, example))
            enc_dec_recon_fold_nodes.append(grassmodel.encode_decode_recon_structure_fold(enc_fold, example))
            enc_dec_label_fold_nodes.append(grassmodel.encode_decode_label_structure_fold(enc_fold, example))

        total_loss = enc_fold.apply(encoder_decoder, [enc_dec_fold_nodes, enc_dec_recon_fold_nodes, enc_dec_label_fold_nodes])
        sum_loss = total_loss[0].sum() / len(batch)
        recon_loss = total_loss[1].sum() / len(batch)
        label_loss = total_loss[2].sum() / len(batch)

#        total_loss = enc_fold.apply(encoder_decoder, [enc_dec_fold_nodes])
#        sum_loss = total_loss[0].sum() / len(batch)

        encoder_decoder_opt.zero_grad()
        sum_loss.backward()
        encoder_decoder_opt.step()

        # Report statistics
        if batch_idx % config.show_log_every == 0:
            print(log_template.format(strftime("%H:%M:%S",time.gmtime(time.time()-start)),
                epoch, config.epochs, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx+len(train_iter)*epoch) / (len(train_iter)*config.epochs),
                learning_rate, label_loss.data[0], recon_loss.data[0], sum_loss.data[0]))
        count=count+1
        if count%2000==1999:
            learning_rate = learning_rate * 0.5
        
    # Save snapshots of the models being trained
    if config.save_snapshot and (epoch+0) % config.save_snapshot_every == 0 :
        print("Saving snapshots of the models ...... ", end='', flush=True)
        torch.save(encoder_decoder, snapshot_folder+'/encoder_decoder_model_epoch_{}_loss_{:.4f}.pkl'.format(epoch+0, sum_loss.data[0]))
        print("DONE")
   
# Save the final models
print("Saving final models ...... ", end='', flush=True)
torch.save(encoder_decoder, config.save_path+'/encoder_decoder_model.pkl')
print("DONE")