import torch
import torch.backends.cudnn as cudnn
import data_loader
import utils
import global_variables as gv
import models.conditional_complex_GRU as complex_GRU
import models.conditional_GRU as complexer_GRU
import models.conditional_multilinear_GRU as multilinear_GRU
import data_loaders.data_loader_conditioned as data_loader
import argparse
import os
import time
from tensorboardX import SummaryWriter
import numpy as np

def setup(network, cuda,device_id=0,wt_load='NIL'):
    if cuda:
        torch.cuda.set_device(device_id)
    train_data_loader = data_loader.dataloader(gv.batch_limit,gv.train_start_index,gv.train_end_index)
    val_data_loader = data_loader.dataloader(gv.batch_limit,gv.val_start_index,gv.val_end_index)
    if network=="complex":
        network = complex_GRU.network(3,gv.hidden_1_size,gv.hidden_2_size,gv.hidden_3_size,gv.output_size)
    elif network=="complexer":
        network = complexer_GRU.network(3,gv.hidden_1_size,gv.hidden_2_size,gv.hidden_3_size,gv.output_size)
    elif network=="multilinear":
        network = multilinear_GRU.network(3,gv.hidden_1_size,gv.hidden_2_size,gv.hidden_3_size,gv.output_size)
    network.cuda()
    # init the network with orthogonal init and gluroot.
    if wt_load=="NIL":
        network.wt_init()
    else:
        pass
        # load wts if required.
    
    # Optimizer 
    optimizer = torch.optim.Adam(network.parameters(), gv.orig_lr, weight_decay=gv.weight_decay)
    
    # for le rumours 
    cudnn.benchmark = True
    
    train_writer = SummaryWriter(os.path.join(gv.tensorboardX_dir,gv.exp_name,'train'))
    val_writer = SummaryWriter(os.path.join(gv.tensorboardX_dir,gv.exp_name,'val'))
    
    return train_data_loader, val_data_loader, network, optimizer,train_writer,val_writer

def train(train_data_loader, network,optimizer,writer,jter_count):
    
    data_fetcher = train_data_loader.get_data()
    count = 0
    for jter,(data,data_gt,characters) in enumerate(data_fetcher):
        cat_target = np.concatenate(data_gt,axis=0)
        x_gt = torch.autograd.Variable(torch.FloatTensor(cat_target[:,1]).cuda())
        y_gt = torch.autograd.Variable(torch.FloatTensor(cat_target[:,2]).cuda())
        pen_down_gt = torch.autograd.Variable(torch.LongTensor(cat_target[:,0]).cuda())
        
        output,loss_distr,pen_loss = network.forward_unlooped_trainer(data,data_gt,characters,optimizer, cuda=True)
        
        
        
        # Summarization
        utils.write_summaries(['data/LossDistr','data/Lossloss'],[loss_distr,pen_loss],[0,0],writer,jter+jter_count)
        
        if jter%gv.update_step ==0:
            print('cur_iter',jter,"cur_loss",loss_distr.cpu().data.numpy(),'batch_size',len(data))
            # add the handwriting pred and gt
            first_seq_len = data[0].shape[0]
            output = output[:first_seq_len]
            
            pen_down_prob,o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr = network.go.get_mixture_coef(output)
            predicted_action = network.go.sample_action(pen_down_prob[:first_seq_len],o_pi[:first_seq_len],
                                                           o_mu1[:first_seq_len], o_mu2[:first_seq_len], 
                                                           o_sigma1[:first_seq_len], o_sigma2[:first_seq_len],
                                                           o_corr[:first_seq_len])
            loss_l2,pen_acc = network.go.val_loss(predicted_action[:first_seq_len], x_gt[:first_seq_len], 
                                                     y_gt[:first_seq_len],pen_down_gt[:first_seq_len])
            pred_image = utils.plot_stroke_numpy(predicted_action.cpu().numpy())# .transpose(2,0,1)
            gt_image = utils.plot_stroke_numpy(data_gt[0])# .transpose(2,0,1)
            
            name_list = ['data/L2Dist','data/PenAcc','train/PredictedSeq','train/GTSeq']
            value_list = [loss_l2,pen_acc,pred_image,gt_image]
            utils.write_summaries(name_list, value_list, [0,0,1,1], writer, jter+jter_count)
    jter_count = jter_count +jter
    return jter_count
            
        
def val(val_data_loader,network,writer,jter_count):
    data_fetcher = val_data_loader.get_data_single()
    loss_list = []
    for jter,(data,data_gt,characters) in enumerate(data_fetcher):
        cat_target = np.concatenate(data_gt,axis=0)
        x_gt = torch.autograd.Variable(torch.FloatTensor(cat_target[:,1]).cuda())
        y_gt = torch.autograd.Variable(torch.FloatTensor(cat_target[:,2]).cuda())
        pen_down_gt = torch.autograd.Variable(torch.LongTensor(cat_target[:,0]).cuda())
        output = network.forward_unlooped(data,characters,cuda=True)
        
        pen_down_prob,o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr = network.go.get_mixture_coef(output)
        loss_distr,pen_loss = network.go.loss_distr(pen_down_prob,o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, x_gt, y_gt,pen_down_gt)
        predicted_action = network.go.sample_action(pen_down_prob,o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr)
        loss_l2,pen_acc = network.go.val_loss(predicted_action, x_gt, y_gt,pen_down_gt)
        
        loss_list.append(loss_l2.cpu().data.numpy())
        
        # Summarization
        name_list = ['data/LossDistr','data/Lossloss','data/L2Dist','data/PenAcc']
        value_list = [loss_distr,pen_loss,loss_l2,pen_acc]
        utils.write_summaries(name_list,value_list, [0]*4, writer, jter+jter_count)
        
        if jter%gv.update_step ==0:
            pred_image = utils.plot_stroke_numpy(predicted_action.cpu().numpy())# .transpose(2,0,1)
            gt_image = utils.plot_stroke_numpy(data_gt[0])# .transpose(2,0,1)
            utils.write_summaries(['val/PredictedSeq','val/GTSeq'],[pred_image,gt_image],[1,1],writer,jter+jter_count)
        
        if jter%gv.update_step ==0:
            print('cur_iter',jter,"cur_loss",loss_l2.cpu().data.numpy(),'loss_pen',pen_acc.cpu().data.numpy())
    loss = sum(loss_list)/float(jter+1)
    print('==========TOTAL VAL LOSS',loss," ====================")
    jter_count = jter+jter_count
        
    return loss,jter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default='True', help='The dataset the class to processed')
    parser.add_argument('--network', default='complexer', help='The dataset the class to processed')
    parser.add_argument('--gpu_id', default='0', help='The dataset the class to processed')
    args = parser.parse_args()
    
    (train_data_loader, val_data_loader, network,optimizer,
        train_writer, test_writer) = setup(args.network, bool(args.cuda),int(args.gpu_id))
    
    # Everything seems fine. 
    # make a code log with exp name
    utils.save_exp_information()
    
    # init values
    train_jter_count,val_jter_count,best_loss = (0,0,np.inf)
    
    for epoch in range(gv.total_epochs):
        train_st_time = time.time()
        utils.adjust_learning_rate(optimizer, epoch,gv.orig_lr)
        train_data_loader.shuffle_index()
        train_jter_count = train(train_data_loader, network, optimizer,train_writer,train_jter_count)
        print('==========TRAIN Epoch',epoch+1,"COMPLETE ====================")
        print('==========TIME TAKEN: ',time.time()-train_st_time,' =============')
        val_st_time = time.time()
        loss,val_jter_count = val(val_data_loader,network,test_writer,train_jter_count)
        print('==========val Epoch',epoch+1,"COMPLETE ====================")
        print('==========TIME TAKEN: ',time.time()-val_st_time,' =============')
        if loss<best_loss:
            print('========== BEST MODEL TILL NOW! =============')
            best_loss_ = True
            best_loss = loss 
        else:
            best_loss_ = False
        # add a is best checker
        utils.save_checkpoint({
               'epoch': epoch + 1,
               'arch': 'res18',
               'loss': loss,
               'model_state_dict': network.state_dict(),
               'optimizer' : optimizer.state_dict(),
            },filename = 'weights/'+gv.exp_name+'_'+str(epoch+1)+'.pth',is_best = best_loss_)
        print('==========Total Time per epoch: ',time.time()-val_st_time,' =============')
    train_writer.close()
    test_writer.close()

if __name__== "__main__":
    main()