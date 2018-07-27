
import numpy as np
import torch


def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_rnn_best.pth.tar')
        
        
        

def adjust_learning_rate(optimizer, epoch,orig_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 3 epochs"""
    lr = orig_lr * (0.1 ** (epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr