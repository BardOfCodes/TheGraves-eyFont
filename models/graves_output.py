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
import torch.distributions.multivariate_normal as mn


    
class network(nn.Module):
    
    def __init__(self,pen_loss_lambda = gv.pen_loss_lambda, pen_loss_weight=gv.pen_loss_weight):
        super(network, self).__init__()
        # add balancing
        self.softmax_loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor(pen_loss_weight))
        self.pen_loss_lambda = pen_loss_lambda
        
        
    def get_mixture_coef(self,output):
        z = output # pen states
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = torch.split(z[:, :-2], gv.output_dist_gaussian_count, 1)
        pen_down_prob = z[:,-2:]

        z_pi = torch.nn.functional.softmax(z_pi)
        # all our actions: limit 40
        z_mu1 =  gv.delta_max*2/math.pi*torch.atan(z_mu1)
        z_mu2 =  gv.delta_max*2/math.pi*torch.atan(z_mu2)

        z_sigma1 = torch.exp(z_sigma1)
        z_sigma2 = torch.exp(z_sigma2)
        z_corr = 0.99* 2/math.pi*torch.atan(z_corr)

        r = [pen_down_prob,z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr]
        return r

    def normal_2d_pdfval(self,x1, x2, mu1, mu2, s1, s2, rho):
        norm1 = x1- mu1
        norm2 = x2- mu2
        s1s2 = s1*s2
        # eq 25
        z = torch.pow(norm1/s1,2) + torch.pow(norm2/s2,2) - 2 * rho*norm1*norm2/s1s2
        neg_rho = 1 - torch.pow(rho,2)
        result = torch.exp(-z/(2 * neg_rho))
        denom = 2 * np.pi * s1s2* torch.sqrt(neg_rho)+1e-20
        result = result/denom
        return result

    def loss_distr(self,pen_down_prob, z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,x1_data, x2_data,pen_data):
        x1_data = torch.stack((x1_data,)*20,1)
        x2_data = torch.stack((x2_data,)*20,1)

        result0 = self.normal_2d_pdfval(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2,z_corr)
        result1 = result0* z_pi
        epsilon = 1e-6
        result1 = torch.sum(result1, 1) + epsilon
        result1 = -torch.log(result1)  # avoid log(0)

        result2 = self.softmax_loss(pen_down_prob,pen_data)
        
        result1 = torch.mean(result1)# + self.pen_loss_weight*result2
        
        return result1,result2
    
    def sample_action(self,pen_down_prob,o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr,no_sigma=False):
        # define action sampling as well
        # pen_out = torch.where(pen_down_prob<0.5, torch.zeros(pen_down_prob.size()[0]).cuda(),torch.ones(pen_down_prob.size()[0]).cuda())
        _, pen_out = torch.sort(pen_down_prob,1,descending=True)
        pen_out = pen_out[:,0:1].float()
        opi_max, opi_sorted = torch.sort(o_pi, 1,descending=True)
        actions = []
        for i in range(opi_max.size()[0]):
            ind = opi_sorted[i,0]
            cur_mu1 = o_mu1[:,ind][i]
            cur_mu2 = o_mu2[:,ind][i]
            cur_sigma1 = o_sigma1[:,ind][i]
            cur_sigma2 = o_sigma2[:,ind][i]
            cur_corr = o_corr[:, ind][i]
            # print(cur_mu1,cur_mu2,cur_sigma1,cur_sigma2,cur_corr)
            sigma_up = [cur_sigma1**2, cur_corr*cur_sigma1*cur_sigma2]
            sigma_down = [cur_corr*cur_sigma1*cur_sigma2, cur_sigma2**2 ]
            if no_sigma:
                sigma = torch.eye(2)#.cuda()
            else:
                sigma = torch.Tensor([sigma_up,sigma_down])
            m = mn.MultivariateNormal(torch.Tensor([cur_mu1,cur_mu2]), sigma)
            actions.append(m.sample().cuda())
        actions = torch.stack(actions,0)
        # print(type(pen_out))
        actions = torch.cat((pen_out,actions),1)
        # print('actions',actions.shape)
        return actions
    
    def val_loss(self,predicted_action, x_gt, y_gt,pen_down_gt):
        
        # L2 of x, and y
        x_diff = torch.mean(torch.norm(predicted_action[:,1:]-torch.stack([x_gt,y_gt],1),2,dim=1))
        # return precision of pen_down_gt
        pen_acc = torch.mean(torch.where(predicted_action[:,0].long() == pen_down_gt,torch.ones(pen_down_gt.size()[0]).cuda(),torch.zeros(pen_down_gt.size()[0]).cuda()))
        
        return x_diff, pen_acc
    
    
