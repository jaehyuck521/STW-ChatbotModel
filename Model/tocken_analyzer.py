
import pandas as pd
import os
file_list = ['chatbot_preprocessed.csv']

train_data = pd.read_csv(file_list[0])
train_data.head()


# In[ ]:


from torchtext.legacy import data, datasets
import os


# # soynlp tokenizer

# In[ ]:




# In[ ]:


import math
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchtext
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# In[ ]:


# Setup seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# trainsformer model 

# In[ ]:
cuda_num = "cuda:0"
model_name = "v3_preprocessed"
output_dir = "./snapshot_" + model_name
from soynlp.tokenizer import LTokenizer


# In[ ]:


tokenizer = LTokenizer()


# In[ ]:


tokenizer("내일 역 앞의 식당에서 밥 먹으러 나갈래 ?")

from collections import Counter
a = []
for line in train_data.values:
    for context in line:
        a += tokenizer(context)
        
c = Counter(a).most_common()
cs = {x[0]:x[1] for x in c}

for i in range(len(train_data)):
    tmp_q = train_data.iloc[i]['Q']
    tmp_a = train_data.iloc[i]['A']
    tmp = tokenizer(tmp_q)
    tmp_res_q = []
    for t in tmp:
        if 300/cs[t] < 1:
            print(t)
        if 300/cs[t] > random.random():
            tmp_res_q.append(t)
            
    tmp = tokenizer(tmp_a)
    tmp_res_a = []
    for t in tmp:
        if 300/cs[t] > random.random():
            tmp_res_a.append(t)
    train_data.iloc[i]['Q'] = " ".join(tmp_res_q)
    train_data.iloc[i]['A'] = " ".join(tmp_res_a)
    
    
    
a = []
for line in train_data.values:
    for context in line:
        a += tokenizer(context)
        
c = Counter(a).most_common()
cs = {x[0]:x[1] for x in c}
    
    


print(c[:250])
