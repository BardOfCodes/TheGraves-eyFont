

################### dataset details
data_loc = 'data/strokes.npy'
char_loc = 'data/sentences.txt'
means = (0.039828256, 0.41248125)
stds = (0.19555554, 2.0786476)
batch_limit = 10000
train_start_index = 0
train_end_index = 5500
val_start_index = 5500
val_end_index = 5750
total_epochs = 30

#################### Network details
hidden_1_size = 400
hidden_2_size = 400
hidden_3_size = 400
output_dist_gaussian_count = 20
output_size = output_dist_gaussian_count*6 +2
delta_max = 40.0
conditional_gaussians = 10

############## optimizer details
weight_decay=0.0001
orig_lr = 0.001
pen_loss_weight = [0.2,1]
pen_loss_lambda = 1
reduction_epoch = 5

############## log details
experiment_code_log_folder = 'logs/code' 
exp_name = "ConditionalComplexBalancedGRU"
update_step = 200
tensorboardX_dir = 'logs/tensorboardX'