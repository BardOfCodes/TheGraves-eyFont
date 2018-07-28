# models script
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.init as init
import time
import os
import random
import math
import torch.nn as nn
import numpy as np
import global_variables as gv
import graves_output
import torch.distributions.multivariate_normal as mn
# import tensorboardX as tb

class network(nn.Module):
    
    def __init__(self, inp_size, hidden_size_1,output_size):
        super(network, self).__init__()
        self.gru_1 = nn.GRU(inp_size, hidden_size_1)
        self.linear = nn.Linear(hidden_size_1,output_size)
        self.hidden_size_1 = hidden_size_1
        self.inp_size = inp_size
        # for training
        self.drop = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        self.go = graves_output.network()
    
    def wt_init(self):
        for name, param in self.named_parameters():
           if 'gru' in name and 'weight' in name:
               init.orthogonal(param);
           elif 'linear' in name:
               init.normal(param,0,math.sqrt(2./float(gv.gru_size+121)))

    def forward_unlooped(self,data, train=True,cuda=True):
        
        seq_lengths = map(len, data)
        limit = max(seq_lengths)
        if cuda:
            seq_tensor = Variable(torch.zeros((limit,len(data),3)).cuda())
        else:
            seq_tensor = Variable(torch.zeros((limit,len(data),3)))
        sorted_ind = np.argsort(seq_lengths)[::-1]
        for idx, (sample, seqlen) in enumerate(zip(data, seq_lengths)):
            if cuda:
                seq_tensor[:seqlen, sorted_ind[idx], :] = torch.FloatTensor(sample).cuda()
            else:
                seq_tensor[:seqlen, sorted_ind[idx], :] = torch.FloatTensor(sample)
    
        seq_lengths.sort(reverse=True)
        pack = torch.nn.utils.rnn.pack_padded_sequence(seq_tensor,seq_lengths)
        if cuda:
            hx = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_1).cuda()) 
        else:
            hx = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_1))   
        
        output = self.gru_1(pack, hx)
        
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output[0])
        unsort_pattrn  = np.argsort(sorted_ind)
        inp_linear = []
        for j in unsort_pattrn:
            inp_linear.append(unpacked[:unpacked_len[j],j,:])
        #print(inp_linear[0].size(),inp_linear[1].size())
        inp_linear = torch.cat(inp_linear,0)
        
        output = self.linear(self.relu(inp_linear))
        return output
    
    def forward_looped(self,batch_size,pred_limit,cuda=True,no_sigma= False):
        if cuda:
            input = torch.autograd.Variable(torch.zeros(1,batch_size, self.inp_size).cuda()) 
            hx = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_1).cuda()) 
        else:
            input = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_1)) 
            hx = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_1)) 
        jter = 0
        actions = []
        while (jter<pred_limit):
            output,hx = self.gru_1(input, hx)
            output = self.linear(self.relu(output[0]))
            pen_down_prob,o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr = self.go.get_mixture_coef(output)
            predicted_action = self.go.sample_action(pen_down_prob,o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr,no_sigma=no_sigma)
            actions.append(predicted_action)
            input = torch.unsqueeze(predicted_action,0)
            jter+=1
        actions = torch.stack(actions,0)
        return actions
        
    


    

