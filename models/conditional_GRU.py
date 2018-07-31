# models script
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.init as init
import time
import os
import sys
sys.path.insert(0,'..')
import random
import math
import torch.nn as nn
import numpy as np
import global_variables as gv
import graves_output
import torch.distributions.multivariate_normal as mn
import conditioning_model as cm
# import tensorboardX as tb

class network(nn.Module):
    
    def __init__(self, inp_size, hidden_size_1,hidden_size_2,hidden_size_3,output_size):
        super(network, self).__init__()
        
        # for the conditioning weights weights
        self.conditioner = cm.conditioner()
        self.condition_len = len(self.conditioner.string)
        self.cond_gauss = gv.conditional_gaussians
        self.gru_1 = nn.GRU(inp_size+self.condition_len, hidden_size_1)
        self.cond_linear = nn.Linear(hidden_size_1, self.cond_gauss*3)
        
        self.gru_2 = nn.GRU(hidden_size_1+self.condition_len, hidden_size_2)
        self.gru_3 = nn.GRU(hidden_size_2+self.condition_len,hidden_size_3)
        self.linear_1 = nn.Linear(hidden_size_3+hidden_size_2+hidden_size_1+ self.condition_len,output_size)
        
        self.inp_size = inp_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_size = output_size
        # for training
        self.drop = nn.Dropout(0.4)
        ## not being used as its not overfitting to train
        self.relu = nn.ReLU()
        self.go = graves_output.network()
    
    def wt_init(self):
        for name, param in self.named_parameters():
           if 'gru' in name and 'weight' in name:
               init.orthogonal(param);
           elif 'linear' in name:
               init.normal(param,0,math.sqrt(2./float(self.hidden_size_1+self.hidden_size_2+self.hidden_size_3)))

        
    def forward_unlooped(self,data, characters, cuda=True,no_sigma= False):
        ## normal Data process
        seq_lengths = map(len, data)
        limit = max(seq_lengths)
        
        if cuda:
            hx_1 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_1).cuda()) 
            hx_2 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_2).cuda()) 
            hx_3 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_3).cuda()) 
            conditional_inp = torch.autograd.Variable(torch.zeros(1,len(data), self.condition_len).cuda()) 
            cond_seq_tensor = Variable(torch.zeros((limit,len(data),3*self.cond_gauss)).cuda())
        else:
            hx = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_1))   
            hx_2 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_2)) 
            hx_3 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_3)) 
            hx_3 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_3)) 
            conditional_inp = torch.autograd.Variable(torch.zeros(1,len(data), self.condition_len)) 
            cond_seq_tensor = Variable(torch.zeros((limit,len(data),3*self.cond_gauss)))
            
        if cuda:
            seq_tensor = Variable(torch.zeros((limit,len(data),3)).cuda())
        else:
            seq_tensor = Variable(torch.zeros((limit,len(data),3)))
        sorted_ind = np.argsort(seq_lengths)[::-1]
        seq_lengths.sort(reverse=True)
        for idx, seqlen in enumerate(seq_lengths):
            sample = data[sorted_ind[idx]]
            if cuda:
                seq_tensor[:seqlen, idx, :] = torch.FloatTensor(sample).cuda()
            else:
                seq_tensor[:seqlen, idx, :] = torch.FloatTensor(sample)
                
        jter=0
        outputs = []
        while (jter<limit):
            inp_1 = torch.cat([seq_tensor[jter:jter+1,:,:],conditional_inp],2)
            output_1,hx_1 = self.gru_1(inp_1, hx_1)
            
            conditional_params = self.cond_linear(output_1[0])
            conditional_params = torch.unsqueeze(conditional_params,0)
            # 1,B,P*3
            tmp_sort_ind = np.array(range(len(characters)))
            conditioning_inp = self.conditioner.get_inp(characters,tmp_sort_ind, conditional_params)
            # 1,B,T
            
            inp_2 = torch.cat([output_1,conditioning_inp],2)
            inp_2 = self.relu(inp_2)
            output_2,hx_2 = self.gru_2(inp_2,hx_2)
            
            inp_3 = torch.cat([output_2,conditioning_inp],2)
            inp_3 = self.relu(inp_3)
            output_3, hx_3 = self.gru_3(inp_3, hx_3)
            
            inp_4 = torch.cat([output_3[0],output_2[0],output_1[0],conditioning_inp[0]],1)
            inp_4 = self.relu(inp_4)
            output = self.linear_1(inp_4)
            
            
            outputs.append(output)
            jter+=1
        outputs = torch.stack(outputs,0)
        # print('Should be S,B,122',outputs.size())
        
        unsort_pattrn  = np.argsort(sorted_ind)
        output = []
        for j in unsort_pattrn:
            output.append(outputs[:seq_lengths[j],j,:])
            
        #print(inp_linear[0].size(),inp_linear[1].size())
        output = torch.cat(output,0)
        # print('should be S*B,122',output.size())
        
        return output
    
    
    def forward_unlooped_trainer(self,data, data_gt, characters, optimizer, cuda=True,no_sigma= False):
        ## normal Data process
        seq_lengths = map(len, data)
        limit = max(seq_lengths)
        
        
        if cuda:
            hx_1 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_1).cuda()) 
            hx_2 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_2).cuda()) 
            hx_3 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_3).cuda()) 
            conditional_inp = torch.autograd.Variable(torch.zeros(1,len(data), self.condition_len).cuda()) 
            cond_seq_tensor = Variable(torch.zeros((limit,len(data),3*self.cond_gauss)).cuda())
        else:
            hx = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_1))   
            hx_2 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_2)) 
            hx_3 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_3)) 
            hx_3 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_3)) 
            conditional_inp = torch.autograd.Variable(torch.zeros(1,len(data), self.condition_len)) 
            cond_seq_tensor = Variable(torch.zeros((limit,len(data),3*self.cond_gauss)))
            
        if cuda:
            seq_tensor = Variable(torch.zeros((limit,len(data),3)).cuda())
            seq_tensor_gt = Variable(torch.zeros((limit,len(data),3)).cuda())
            seq_tensor_wt = Variable(torch.zeros((limit,len(data))).cuda())
        else:
            seq_tensor = Variable(torch.zeros((limit,len(data),3)))
            seq_tensor_gt = Variable(torch.zeros((limit,len(data),3)))
            seq_tensor_wt = Variable(torch.zeros((limit,len(data))))
            
        sorted_ind = np.argsort(seq_lengths)[::-1]
        seq_lengths.sort(reverse=True)
        for idx, seqlen in enumerate(seq_lengths):
            sample = data[sorted_ind[idx]]
            sample_gt = data_gt[sorted_ind[idx]]
            if cuda:
                seq_tensor[:seqlen, idx, :] = torch.FloatTensor(sample).cuda()
                seq_tensor_gt[:seqlen, idx, :] = torch.FloatTensor(sample_gt).cuda()
            else:
                seq_tensor[:seqlen, idx, :] = torch.FloatTensor(sample)
                seq_tensor_gt[:seqlen, idx, :] = torch.FloatTensor(sample_gt)
            seq_tensor_wt[:seqlen, idx ] = 1
                
        jter=0
        outputs = []
        loss_distr_list = []
        loss_pen_list = []
        while (jter<limit):
            cur_wt = seq_tensor_wt[jter,:]
            pen_down_gt = seq_tensor_gt[jter,:,0].long()
            x_gt = seq_tensor_gt[jter,:,1]
            y_gt = seq_tensor_gt[jter,:,2]
            
            inp_1 = torch.cat([seq_tensor[jter:jter+1,:,:],conditional_inp],2)
            output_1,hx_1 = self.gru_1(inp_1, hx_1)
            
            conditional_params = self.cond_linear(output_1[0])
            conditional_params = torch.unsqueeze(conditional_params,0)
            # 1,B,P*3
            tmp_sort_ind = np.array(range(len(characters)))
            conditioning_inp = self.conditioner.get_inp(characters,tmp_sort_ind, conditional_params)
            # 1,B,T
            
            inp_2 = torch.cat([output_1,conditioning_inp],2)
            inp_2 = self.relu(inp_2)
            output_2,hx_2 = self.gru_2(inp_2,hx_2)
            
            inp_3 = torch.cat([output_2,conditioning_inp],2)
            inp_3 = self.relu(inp_3)
            output_3, hx_3 = self.gru_3(inp_3, hx_3)
            
            inp_4 = torch.cat([output_3[0],output_2[0],output_1[0],conditioning_inp[0]],1)
            inp_4 = self.relu(inp_4)
            output = self.linear_1(inp_4)
            
            #print(output.size(),x_gt.size(),y_gt.size(),pen_down_gt.size(),cur_wt.size())
            pen_down_prob,o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr = self.go.get_mixture_coef(output)
            loss_distr,pen_loss = self.go.loss_distr(pen_down_prob,o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, x_gt, y_gt,pen_down_gt,loss_mean=False)
            #print(loss_distr.size(),pen_loss.size())
            loss_distr = torch.mean(torch.mul(loss_distr,cur_wt)) 
            pen_loss = torch.mean(torch.mul(pen_loss,cur_wt))
            loss_distr_list.append(loss_distr)
            loss_pen_list.append(pen_loss)
            total_loss = loss_distr +pen_loss
            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            # grad clipping
            # for name, param in network.named_parameters():
            #     if 'gru' in name:
            #         param.grad.clamp_(-10, 10)
            #     elif 'linear' in name:
            #         param.grad.clamp_(-100, 100)
            optimizer.step()
            hx_1 = Variable(hx_1.data)#clone()
            hx_2 = Variable(hx_2.data)#clone()
            hx_3 = Variable(hx_3.data)#clone()
            conditional_inp = Variable(conditional_inp.data)#clone()
            
            outputs.append(output)
            jter+=1
        outputs = torch.stack(outputs,0)
        loss_distr = torch.mean(torch.stack(loss_distr_list,0))
        pen_loss = torch.mean(torch.stack(loss_pen_list,0))
        # print('Should be S,B,122',outputs.size())
        
        unsort_pattrn  = np.argsort(sorted_ind)
        output = []
        for j in unsort_pattrn:
            output.append(outputs[:seq_lengths[j],j,:])
            
        #print(inp_linear[0].size(),inp_linear[1].size())
        output = torch.cat(output,0)
        # print('should be S*B,122',output.size())
        
        return output,loss_distr,pen_loss
    
    
    def forward_looped(self,characters,pred_limit,cuda=True,no_sigma= False):
        batch_size = len(characters)
        if cuda:
            inp_1 = torch.autograd.Variable(torch.zeros(1,batch_size, self.inp_size).cuda()) 
            hx_1 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_1).cuda()) 
            hx_2 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_2).cuda()) 
            hx_3 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_3).cuda()) 
            conditional_inp = torch.autograd.Variable(torch.zeros(1,batch_size, self.condition_len).cuda()) 
        else:
            inp_1 = torch.autograd.Variable(torch.zeros(1,batch_size, self.inp_size)) 
            hx_1 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_1)) 
            hx_2 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_2)) 
            hx_3 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_3)) 
            inp_1 = torch.autograd.Variable(torch.zeros(1,batch_size, self.inp_size)) 
            conditional_inp = torch.autograd.Variable(torch.zeros(1,batch_size, self.condition_len)) 
        jter = 0
        actions = []
        while (jter<pred_limit):
            inp_1 = torch.cat([inp_1,conditional_inp],2)
            output_1,hx_1 = self.gru_1(inp_1, hx_1)
            
            conditional_params = self.cond_linear(output_1[0])
            conditional_params = torch.unsqueeze(conditional_params,0)
            # 1,B,P*3
            sorted_ind = np.array(range(len(characters)))
            conditioning_inp = self.conditioner.get_inp(characters,sorted_ind, conditional_params)
            # 1,B,T
            
            inp_2 = torch.cat([output_1,conditioning_inp],2)
            inp_2 = self.relu(inp_2)
            output_2,hx_2 = self.gru_2(inp_2,hx_2)
            
            inp_3 = torch.cat([output_2,conditioning_inp],2)
            inp_3 = self.relu(inp_3)
            output_3, hx_3 = self.gru_3(inp_3, hx_3)
            
            inp_4 = torch.cat([output_3[0],output_2[0],output_1[0],conditioning_inp[0]],1)
            inp_4 = self.relu(inp_4)
            output = self.linear_1(inp_4)
            
            pen_down_prob,o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr = self.go.get_mixture_coef(output)
            predicted_action = self.go.sample_action(pen_down_prob,o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr,no_sigma=no_sigma)
            actions.append(predicted_action)
            inp_1 = torch.unsqueeze(predicted_action,0)
            jter+=1
        actions = torch.stack(actions,0)
        return actions
    
        
def test():
    torch.cuda.set_device(0)
    data = [np.random.uniform(size=(500,3)),np.random.uniform(size=(556,3)),np.random.uniform(size=(226,3))]
    net = network(3,256,512,256,122)
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), 0.001, weight_decay=0.0001)
    char_arr = ['asdfjh','askdfjhaksdfj','asdfasdfasdfasdf']
    # output = net.forward_unlooped(data,char_arr,cuda=True)
    # print('Unlooped Done!',output.size())
    # # output = net.forward_looped(char_arr, 700)
    # print("Looped Done!",output.size())
    output = net.forward_unlooped_trainer(data,data, char_arr, optimizer, cuda=True,no_sigma= False)
    print('training!')
    
    
if __name__ == '__main__':
    test()

    

