

################### dataset details
data_loc = 'data/strokes.npy'
means = (0.039828256, 0.41248125)
stds = (0.19555554, 2.0786476)
batch_limit = 3000
train_start_index = 0
train_end_index = 5000
val_start_index = 5000
val_end_index = 5400
total_epochs = 10

#################### Network details
gru_size = 900
output_dist_gaussian_count = 20
output_dim = output_dist_gaussian_count*5 +1
delta_max = 40.0

############## optimizer details
weight_decay=0.0001
orig_lr = 0.001
pen_loss_weight = 1

############## log details
experiment_code_log_folder = 'logs/code' 
exp_name = "SimpleGRU"
update_step = 50
tensorboardX_dir = 'logs/tensorboardX'