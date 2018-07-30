import torch
import numpy as np
import global_variables as gv
class conditioner:
    
    def __init__(self): 
        self.string = " abcdefghijklmnopqrstuvwxyzABcdefghijklmnopqrstuvwxyz.@"
        self.char_int_map = {self.string[i]:i for i in range(len(self.string))}
        self.cond_gaussians = gv.conditional_gaussians
    def char_to_int(self,sample):
        int_repr = []
        for xx in sample:
            if xx in self.string:
                int_repr.append(self.char_int_map[xx])
            else:
                int_repr.append(self.char_int_map['@'])
        return int_repr
    
    def int_to_char(self,int_repr):
        char_repr = []
        for inter in int_repr:
            char_repr.append(self.string[inter])
        return ''.join(char_repr)
    
    def char_to_one_hot(self,sample,cuda=True):
        int_repr = self.char_to_int(sample)
        int_repr = np.array(int_repr)[:,None]
        if cuda:
            y = torch.LongTensor(int_repr).cuda()
            y_onehot = torch.FloatTensor(int_repr.shape[0],len(self.string)).cuda()
        else:
            y = torch.LongTensor(int_repr)
            y_onehot = torch.FloatTensor(int_repr.shape[0],len(self.string))  
        y_onehot.zero_()
        output = y_onehot.scatter_(1, y, 1)
        # U,T
        return output
    
    def get_char_tensor(self,char_array,sorted_ind, cuda = True):
        
        lens = map(len, char_array)
        if cuda:
            char_tensor = torch.FloatTensor(len(char_array),max(lens),len(self.string)).cuda()
        else:
            char_tensor = torch.FloatTensor(len(char_array),max(lens),len(self.string))
        char_tensor.zero_()
        for i in range(len(lens)):
            sample = self.char_to_one_hot(char_array[sorted_ind[i]])
            char_tensor[i,:lens[sorted_ind[i]],:] = sample
            
        return char_tensor
        
    def get_inp(self,char_array, sorted_ind, cond_seq_tensor, cuda=True):
        # char_arrya are of size [10,20,...]
        # params are S,B,3*P
        # TO make of each of S,B,i get max(U) windows
        # now the Max U will interact with the char array
        # char_tensor is B,U,T
        # now from params make S,B,U 
        cond_seq_tensor = torch.exp(cond_seq_tensor)
        alphas = cond_seq_tensor[:,:,self.cond_gaussians*0:self.cond_gaussians*1]
        betas = cond_seq_tensor[:,:,self.cond_gaussians*1:self.cond_gaussians*2]
        kappas = cond_seq_tensor[:,:,self.cond_gaussians*2:self.cond_gaussians*3]
        kappas = torch.cumsum(kappas,0)
        
        lens = map(len, char_array)
        char_tensor = self.get_char_tensor(char_array,sorted_ind,cuda) 
        if cuda:
                range_arr = torch.arange(max(lens)).float().cuda()
        else:
            range_arr = torch.arange(max(lens)).float()
        range_arr = torch.stack([range_arr,]*self.cond_gaussians, 0)
        range_arr = torch.stack([torch.stack([range_arr,]*len(char_array),0)]*alphas.size()[0],0)
        # now for all the gaussians we want to do it
        
        # range_arr = S,B,P,U kappas is S,B,P
        kappas = torch.stack([kappas,]*max(lens),3)
        betas = torch.stack([betas,]*max(lens),3)
        alphas = torch.stack([alphas,]*max(lens),3)
        out = torch.exp(torch.mul(torch.pow(kappas - range_arr,2),-betas))
        out = torch.mul(out,alphas)
        out = torch.sum(out,2)
        # should be S,B,U Now we have char of B,U,T
        char_tensor = torch.stack([char_tensor,]*alphas.size()[0],0)
        out = torch.stack([out,]*len(self.string), 3)
        final_out = torch.mul(char_tensor,out)
        final_out = torch.sum(final_out,2)
        # it is S,B,T
        return final_out
    
        
        
        
        