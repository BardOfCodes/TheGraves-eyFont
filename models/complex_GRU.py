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
# import tensorboardX as tb

class network(nn.Module):
    
    def __init__(self, inp_size, hidden_size_1,hidden_size_2,hidden_size_3,output_size):
        super(network, self).__init__()
        self.gru_1 = nn.GRU(inp_size, hidden_size_1)
        self.gru_2 = nn.GRU(hidden_size_1+inp_size, hidden_size_2)
        self.linear_1 = nn.Linear(hidden_size_2+inp_size,hidden_size_3)
        self.linear_2 = nn.Linear(hidden_size_3+hidden_size_2+hidden_size_1,output_size)
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
           elif 'linear_1' in name:
               init.normal(param,0,math.sqrt(2./float(self.hidden_size_2+self.hidden_size_3)))
           elif 'linear_3' in name:
               init.normal(param,0,math.sqrt(2./float(self.hidden_size_1+self.hidden_size_2+self.hidden_size_3)))

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
        inp_1 = torch.nn.utils.rnn.pack_padded_sequence(seq_tensor,seq_lengths)
        if cuda:
            hx_1 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_1).cuda()) 
            hx_2 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_2).cuda()) 
        else:
            hx = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_1))   
            hx_2 = torch.autograd.Variable(torch.zeros(1,len(data), self.hidden_size_2)) 
        
        ###
        hidden_1 = self.gru_1(inp_1, hx_1)
        
        unpacked_1, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(hidden_1[0])
        ########################## 2nd GRU
        inp_2 = torch.cat([unpacked_1,seq_tensor],2)
        inp_2 = torch.nn.utils.rnn.pack_padded_sequence(inp_2,seq_lengths)
        hidden_2 = self.gru_2(inp_2, hx_2)
        unpacked_2, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(hidden_2[0])
        
        unsort_pattrn  = np.argsort(sorted_ind)
        base_inp = []
        hidden_1 = []
        hidden_2 = []
        for j in unsort_pattrn:
            hidden_2.append(unpacked_2[:unpacked_len[j],j,:])
            hidden_1.append(unpacked_1[:unpacked_len[j],j,:])
            base_inp.append(seq_tensor[:unpacked_len[j],j,:])
        #print(inp_linear[0].size(),inp_linear[1].size())
        hidden_1 = torch.cat(hidden_1,0)
        hidden_2 = torch.cat(hidden_2,0)
        base_inp = torch.cat(base_inp,0)
        
        inp_3 = torch.cat([hidden_2,base_inp],1)
        
        hidden_3 = self.relu(self.linear_1(inp_3))
        
        inp_4 = torch.cat([hidden_3,hidden_2,hidden_1],1)       
        
        output = self.linear_2(inp_4)
        
        return output
    
    def forward_looped(self,batch_size,pred_limit,cuda=True,no_sigma= False):
        if cuda:
            inp_1 = torch.autograd.Variable(torch.zeros(1,batch_size, self.inp_size).cuda()) 
            hx_1 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_1).cuda()) 
            hx_2 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_2).cuda()) 
        else:
            inp_1 = torch.autograd.Variable(torch.zeros(1,batch_size, self.inp_size)) 
            hx_1 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_1)) 
            hx_2 = torch.autograd.Variable(torch.zeros(1,batch_size, self.hidden_size_2)) 
        jter = 0
        actions = []
        while (jter<pred_limit):
            output_1,hx_1 = self.gru_1(inp_1, hx_1)
            inp_2 = torch.cat([output_1,inp_1],2)
            output_2,hx_2 = self.gru_2(inp_2,hx_2)
            inp_3 = torch.cat([output_2[0],inp_1[0]],1)
            output_3 = self.relu(self.linear_1(inp_3))
            inp_4 = torch.cat([output_3,output_2[0],output_1[0]],1)
            output = self.linear_2(inp_4)
            
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
    output = net.forward_unlooped(data,cuda=True)
    print('Unlooped Done!',output.size())
    output = net.forward_looped(5,700)
    print("Looped Done!",output.size())
    
    
if __name__ == '__main__':
    test()

    

