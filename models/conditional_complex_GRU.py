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
        self.gru_1 = nn.GRU(inp_size, hidden_size_1)
        
        # for the conditioning weights weights
        self.cond_gauss = gv.conditional_gaussians
        self.cond_linear = nn.Linear(hidden_size_1, self.cond_gauss*3)
        self.conditioner = cm.conditioner()
        self.condition_len = len(self.conditioner.string)
        
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

        
    def forward_unlooped(self,data,characters, train=True,cuda=True, get_encoding=False):
        
        ## normal Data process
        seq_lengths = map(len, data)
        limit = max(seq_lengths)
        
        if cuda:
            hx_1 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_1).cuda()) 
            hx_2 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_2).cuda()) 
            hx_3 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_3).cuda()) 
            cond_seq_tensor = Variable(torch.zeros((limit,len(data),3*self.cond_gauss)).cuda())
        else:
            hx = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_1))   
            hx_2 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_2)) 
            hx_3 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_3)) 
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
        
    
        
        ########################## 1st GRU
        inp_1 = torch.nn.utils.rnn.pack_padded_sequence(seq_tensor,seq_lengths)
        hidden_1 = self.gru_1(inp_1, hx_1)
        unpacked_1, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(hidden_1[0])
        
        ######################### Conditioning        
        unsort_pattrn  = np.argsort(sorted_ind)
        conditional_inp = []
        for j in unsort_pattrn:
            conditional_inp.append(unpacked_1[:unpacked_len[j],j,:])
        conditional_inp = torch.cat(conditional_inp,0)
        
        conditional_params = self.cond_linear(conditional_inp)
        
        # now put them in seq
        temp_seq_lengths = map(len, data)
        temp_seq_lengths.insert(0,0)
        temp_seq_lengths = np.cumsum(temp_seq_lengths)
        params_list = []
        for idx in range(len(temp_seq_lengths)-1):
            params_list.append(conditional_params[temp_seq_lengths[idx]:temp_seq_lengths[idx+1]])
        for idx, seqlen in enumerate(seq_lengths):
            sample = params_list[sorted_ind[idx]]
            cond_seq_tensor[:seqlen, idx, :] = sample
            
        # now cond_seq_tensor should be S,B,3*p
        # Now kappas can be cumalated
        
        conditioning_inp = self.conditioner.get_inp(characters,sorted_ind, cond_seq_tensor)
        if get_encoding:
            return conditioning_inp
        
        ########################## 2nd GRU
        inp_2 = torch.cat([unpacked_1,conditioning_inp],2)
        inp_2 = self.relu(inp_2)
        inp_2 = torch.nn.utils.rnn.pack_padded_sequence(inp_2,seq_lengths)
        hidden_2 = self.gru_2(inp_2, hx_2)
        unpacked_2, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(hidden_2[0])
        
        ######################### 3rd GRU
        inp_3 = torch.cat([unpacked_2,conditioning_inp],2)
        inp_3 = self.relu(inp_3)
        inp_3 = torch.nn.utils.rnn.pack_padded_sequence(inp_3,seq_lengths)
        hidden_3 = self.gru_3(inp_3, hx_3)
        unpacked_3, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(hidden_3[0])
        
        unsort_pattrn  = np.argsort(sorted_ind)
        base_inp = []
        hidden_1 = []
        hidden_2 = []
        hidden_3 = []
        final_conditional = []
        for j in unsort_pattrn:
            hidden_3.append(unpacked_3[:unpacked_len[j],j,:])
            hidden_2.append(unpacked_2[:unpacked_len[j],j,:])
            hidden_1.append(unpacked_1[:unpacked_len[j],j,:])
            final_conditional.append(conditioning_inp[:unpacked_len[j],j,:])
            
        #print(inp_linear[0].size(),inp_linear[1].size())
        hidden_1 = torch.cat(hidden_1,0)
        hidden_2 = torch.cat(hidden_2,0)
        hidden_3 = torch.cat(hidden_3,0)
        final_conditional = torch.cat(final_conditional,0)
        
        ############################ 4th Linear
        inp_4 = torch.cat([hidden_3,hidden_2,hidden_1,final_conditional],1)
        inp_4 = self.relu(inp_4)
        output = self.linear_1(inp_4)
        
        return output
    
    def forward_looped(self,characters,pred_limit,cuda=True,no_sigma= False):
        batch_size = len(characters)
        if cuda:
            inp_1 = torch.autograd.Variable(torch.zeros(1,batch_size, self.inp_size).cuda()) 
            hx_1 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_1).cuda()) 
            hx_2 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_2).cuda()) 
            hx_3 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_3).cuda()) 
        else:
            inp_1 = torch.autograd.Variable(torch.zeros(1,batch_size, self.inp_size)) 
            hx_1 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_1)) 
            hx_2 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_2)) 
            hx_3 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_3)) 
        jter = 0
        actions = []
        while (jter<pred_limit):
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
    char_arr = ['asdfjh','askdfjhaksdfj','asdfasdfasdfasdf']
    output = net.forward_unlooped(data,char_arr,cuda=True)
    print('Unlooped Done!',output.size())
    output = net.forward_looped(char_arr, 700)
    print("Looped Done!",output.size())
    
    
if __name__ == '__main__':
    test()

    

