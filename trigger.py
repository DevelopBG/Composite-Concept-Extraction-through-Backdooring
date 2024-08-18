import torch
import numpy as np


random_seed = 99 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#div' = 2 gives check board, 'div'=1 gives solid

def trigger_checkerboard(x1,div=2, m=3, t = 1., all=True ):
    '''original trigger function'''
    n = m
   
    if all: # adding triger on all samples while keeping label at groung truth
        x = np.copy(x1)
        x = torch.tensor(x)
        # x = x.permute(0, 2, 3, 1).to(device)
        device = 'cpu'
        for i in range(m):
            for k in range(n): 
                if (i+k) % div==0:        ## RGB - now G and B pattern      
                    x[:,0 + i, 0 + k,:] = torch.tensor([1., 0., 0.]).to(device) * t + x[:,0 + i, 0 + k,:] * (1 - t)
                else:
                    x[:,0 + i, 0 + k,:] = torch.tensor([0., 1., 0.]).to(device) * t + x[:,0 + i, 0 + k,:] * (1 - t)
    return torch.clip(x, min=0., max=1)

