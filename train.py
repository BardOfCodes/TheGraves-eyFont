import torch
import torch.backends.cudnn as cudnn
import data_loader
import utils
import global_variables as gv
import models
import argparse
import time
# integrate tensorboardX
import tensorboardX
import numpy as np

def setup(cuda,device_id=0):
    if cuda:
        torch.cuda.set_device(device_id)
    train_data_loader = data_loader.dataloader(gv.batch_limit,gv.train_start_index,gv.train_end_index)
    val_data_loader = data_loader.dataloader(gv.batch_limit,gv.val_start_index,gv.val_end_index)
    network = models.simple_GRU(3,gv.gru_size,121)
    network.cuda()
    # init the network with orthogonal init and gluroot.
    network.wt_init()
    
    graves_output = models.graves_output()
    
    # Optimizer 
    optimizer = torch.optim.Adam(network.parameters(), gv.orig_lr, weight_decay=gv.weight_decay)
    
    # for le rumours 
    cudnn.benchmark = True
    
    return train_data_loader, val_data_loader, network, graves_output, optimizer

def train(train_data_loader, network, graves_output,optimizer):
    
    data_fetcher = train_data_loader.get_data()
    for jter,(data,data_gt) in enumerate(data_fetcher):
        cat_target = np.concatenate(data_gt,axis=0)
        x_gt = torch.autograd.Variable(torch.FloatTensor(cat_target[:,1]).cuda())
        y_gt = torch.autograd.Variable(torch.FloatTensor(cat_target[:,2]).cuda())
        pen_down_gt = torch.autograd.Variable(torch.FloatTensor(cat_target[:,0]).cuda())
        output = network.forward_unlooped(data,cuda=True)
        pen_down_prob,o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr = graves_output.get_mixture_coef(output)
        loss_distr = graves_output.loss_distr(pen_down_prob,o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, x_gt, y_gt,pen_down_gt)
        
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_distr.backward()
        optimizer.step()
        
        if jter%gv.update_step ==0:
            print('cur_iter',jter,"cur_loss",loss_distr.cpu().data.numpy(),'batch_size',len(data))
            
        
def val(val_data_loader,network,graves_output):
    data_fetcher = val_data_loader.get_data_single()
    loss_list = []
    for jter,(data,data_gt) in enumerate(data_fetcher):
        cat_target = np.concatenate(data_gt,axis=0)
        x_gt = torch.autograd.Variable(torch.FloatTensor(cat_target[:,1]).cuda())
        y_gt = torch.autograd.Variable(torch.FloatTensor(cat_target[:,2]).cuda())
        pen_down_gt = torch.autograd.Variable(torch.FloatTensor(cat_target[:,0]).cuda())
        output = network.forward_unlooped(data,cuda=True)
        pen_down_prob,o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr = graves_output.get_mixture_coef(output)
        predicted_action = graves_output.sample_action(pen_down_prob,o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr)
        loss_l2,loss_pen = graves_output.val_loss(predicted_action, x_gt, y_gt,pen_down_gt)
        loss_list.append(loss_l2.cpu().data.numpy())
        if jter%gv.update_step ==0:
            print('cur_iter',jter,"cur_loss",loss_l2.cpu().data.numpy(),'loss_pen',loss_pen.cpu().data.numpy())
    loss = sum(loss_list)/float(jter+1)
        
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default='True', help='The dataset the class to processed')
    parser.add_argument('--gpu_id', default='0', help='The dataset the class to processed')
    args = parser.parse_args()
    
    train_data_loader, val_data_loader, network, graves_output,optimizer = setup(bool(args.cuda),int(args.gpu_id))
    for epoch in range(gv.total_epochs):
        train_st_time = time.time()
        utils.adjust_learning_rate(optimizer, epoch,gv.orig_lr)
        train_data_loader.shuffle_index()
        train(train_data_loader, network, graves_output,optimizer)
        print('==========TRAIN Epoch',epoch+1,"COMPLETE ====================")
        print('==========TIME TAKEN: ',time.time()-train_st_time,' =============')
        val_st_time = time.time()
        loss = val(val_data_loader,network,graves_output)
        print('==========val Epoch',epoch+1,"COMPLETE ====================")
        print('==========TIME TAKEN: ',time.time()-val_st_time,' =============')
        
        # add a is best checker
        utils.save_checkpoint({
               'epoch': epoch + 1,
               'arch': 'res18',
               'loss': loss,
               'model_state_dict': network.state_dict(),
               'optimizer' : optimizer.state_dict(),
            },filename = 'weights/simple_GRU_'+str(epoch+1)+'.pth')
        print('==========Total Time per epoch: ',time.time()-val_st_time,' =============')

if __name__== "__main__":
    main()