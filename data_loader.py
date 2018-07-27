# dataset_script

# with only down strokes 
# ('the data has mean', (0.0422648, 0.56823623), 'and std', (0.20119265, 1.120318))
# with all strokes:
# ('the data has mean', (0.039828256, 0.41248125), 'and std', (0.19555554, 2.0786476))
# In the paper, the data is normalized first.
import global_variables as gv
import numpy as np
import random

class dataloader():
    
    def __init__(self,batch_limit,data_start, data_end, data_loc=gv.data_loc):
        self.all_data = np.load(data_loc)
        self.all_data = [self.all_data[i] for i in range(data_start,data_end)]
        self.means = gv.means
        self.stds = gv.stds
        self.shuffled_index = range(len(self.all_data))
        random.shuffle(self.shuffled_index)
        self.all_data = [self.all_data[i] for i in self.shuffled_index]
        self.batch_limit = batch_limit
        
    def shuffle_index(self):
        random.shuffle(self.shuffled_index)
        self.all_data = [self.all_data[i] for i in self.shuffled_index]
        
    def get_next_count(self,data,new_ar):
        new_data = [x for x in data]
        new_data.append(new_ar)
        size = [len(x) for x in new_data]
        total = max(size)*len(new_data)
        #print(total)
        return total
    
    def get_data(self):
        cur_data_iter = 0
        epoch_size = len(self.all_data)
        while(cur_data_iter<epoch_size):
#             'returned here')
            data = []
            count = 0
            while(self.get_next_count(data,self.all_data[cur_data_iter])<self.batch_limit):
                data.append(self.all_data[cur_data_iter])
                cur_data_iter +=1
                if cur_data_iter>=epoch_size: break
            data_inp = [np.concatenate([np.zeros((1,3)), x[:-1]],0) for x in data]
            yield data_inp,data
            
    def get_data_single(self):
        cur_data_iter = 0
        epoch_size = len(self.all_data)
        while(cur_data_iter<epoch_size):
            data = []
            count = 0
            data.append(self.all_data[cur_data_iter])
            cur_data_iter +=1
            data_inp = [np.concatenate([np.zeros((1,3)), x[:-1]],0) for x in data]
            yield data_inp,data
        

            
        
