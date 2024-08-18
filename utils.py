import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import colorsys
from sklearn.metrics import roc_auc_score


random_seed = 90 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Num_classes(name):
    if name =='gtsrb':
        num_classes =43
    elif name =='svhn':
            num_classes = 10
    elif name =='cifar10':
            num_classes = 10
    elif name =='timagenet':
            num_classes = 200
    elif name =='ytfa':
            num_classes = 1283

    return num_classes


def trigger_1(x1,fix= False, m=2, n=2, t=1.):
    '''
    method to create the full size trigger 
    '''

    x = x1.clone()
    # x = torch.zeros(x1.shape).to(device)

    x = x.permute(0,2,3,1).to(device)
    # x = torch.zeros(x.shape).to(device)
    
    def mask_sq(pos_i=0,pos_j=0):
        x_patch = torch.zeros(x.shape[1:]).to(device)
        t_mask = torch.zeros(x[0].shape).to(device)
        # print(t)
        for i in range(m):
            for k in range(n):

                if (i+k)%2 == 0:
                    t_mask[pos_i+i,pos_j+k] = torch.tensor([1.,1.,1.]).to(device)
                    x_patch[pos_i+i,pos_j+k] = torch.tensor([1.,0.,0.]).to(device)
                else:
                    t_mask[pos_i+i,pos_j+k] = torch.tensor([1.,1.,1.]).to(device)
                    x_patch[pos_i+i,pos_j+k] = torch.tensor([0.,1.,0.]).to(device)
        return t_mask, x_patch
    
    
    for l in range(len(x)):
        torch.manual_seed(l)
        np.random.seed(l)
        
        
        x_i = 2
        x_j = 2
        # t0 = np.random.uniform(0.2, 0.99, len(x))
        # t00 = np.random.uniform(t, 0.2, len(x))

        # t_mask, x_patch = mask_sq(x_i,x_j)
        
        
        t_mask, x_patch = mask_sq(x_i,x_j)

        if not fix:
            if l%2==0:
                # t1 = 1-t0[l]
                t1 = t
            else:
                # t1 = 1-t00[l]
                t1 = t
            x[l] = x[l].to(device) * (1-t_mask*(1-t1)) + x_patch.to(device) * (1-t1)

        else:
            t1 = t
            x[l] = x[l].to(device) * (1-t_mask*(1-t1)) + x_patch.to(device) * (1-t1)
    return torch.clip(x.permute(0,3,1,2),min=0., max=1,)




def trigger_2(x1,fix= True, m=2, n=2, t=1.):
    '''
    making trigger from dataset samples not for dataloader samples
    '''

    x = np.copy(x1)
    x = torch.tensor(x)
    device = 'cpu'
    def mask_sq(pos_i=0,pos_j=0):
        x_patch = torch.zeros(x.shape[1:]).to(device)
        t_mask = torch.zeros(x[0].shape).to(device)
        # print(t)
        for i in range(m):
            for k in range(n):

                if (i+k)%2 == 0:
                    t_mask[pos_i+i,pos_j+k] = torch.tensor([1,1,1]).to(device)
                    x_patch[pos_i+i,pos_j+k] = torch.tensor([255,0,0]).to(device) ## red [255,0,0]
                else:
                    t_mask[pos_i+i,pos_j+k] = torch.tensor([1,1,1]).to(device)
                    x_patch[pos_i+i,pos_j+k] = torch.tensor([0,255,0]).to(device)
        return t_mask, x_patch
    
    
    for l in range(len(x)):
        torch.manual_seed(l)
        np.random.seed(l)
       
        x_i = 2
        x_j = 2
       
        t_mask, x_patch = mask_sq(x_i,x_j)

        if not fix:
            if l%2==0:
                # t1 = 1-t0[l]
                t1 = t
            else:
                # t1 = 1-t00[l]
                t1 = t
            x[l] = x[l].to(device) * (1-t_mask*(1-t1)) + x_patch.to(device) * (1-t1)

        else:
            t1 = t
            x[l] = x[l].to(device) * (1-t_mask*(1-t1)) + x_patch.to(device) * (1-t1)

    return torch.clip(x,min=0, max=255)







def add_perturbation(x,eps):
    x = x.permute(0,2,3,1).to(device)
    for i in range(len(x)):
        delta_x = torch.tensor(np.random.uniform(eps/2,eps,x[i].shape)).to(device)
        x[i] = x[i] + delta_x
    return torch.clip(x,min=0.,max=1.)


def model_saving(model, path):
    torch.save(model.state_dict(),path)

def accuracy_model_test(dataloader, model, m=0,n=0,t=0.,ratio=0.6,target=0,
                   trig=False):
    model.eval()
    count = 0
    samples = 0
    for img, label in dataloader:
        img = img.to(device)
        if trig:
            total_list = np.arange(len(label))
            poison_list = np.random.choice(total_list, int(len(label) * 1),replace= False)
            label[poison_list] = target
            img[poison_list] = trigger_1(img[poison_list],fix = True, m=m, n=n, t=t)
            x = img.to(device)
        else:
            x = img.clone()
        
        label = label.to(device)
        score = model(x)
        probability, pred = score.max(1)
        count += (pred == label).sum()
        samples += len(label)
    return (count / samples) * 100

def accuracy_model_all(dataloader, model,pnt=False):
    model.eval()
    count = 0
    samples = 0
    for img, label in dataloader:
        img = img.to(device)       
        label = label.to(device)
        score = model(img)
        _ , pred = score.max(1)
        count += (pred == label).sum()
        samples += len(label)
        if pnt:
            print(label)
            print(pred)
    model.train()
    return (count / samples) * 100

def accuracy_model(dataloader, model, m=0,n=0,t=0.,ratio=0.6,target=0,
                   trig=False):
    model.eval()
    count = 0
    samples = 0
    for img, label in dataloader:
        img = img.to(device)
        if trig:
            total_list = np.arange(len(label))
            poison_list = np.random.choice(total_list, int(len(label) * 1),replace= False)
            label[poison_list] = target
            img[poison_list] = trigger_1(img[poison_list],fix = True, m=m, n=n, t=t)
            x = img.to(device)
        else:
            x = img.clone()
        
        label = label.to(device)
        score = model(x)
        _, pred = score.max(1)
        count += (pred == label).sum()
        samples += len(label)
    model.train()
    return (count / samples) * 100


def accuracy_train_pure(traindataloader, testdataloader, model):

    model.eval() ## TO EVAL MODE
    trcount = 0
    trsamples = 0
    for x, y in traindataloader:
        x = x.to(device)
        y = y.to(device)
        score = model(x)
        _, pred = score.max(1)
        trcount += (pred == y).sum()
        trsamples += len(y)

    tscount = 0
    tssamples = 0
    for x, y in testdataloader:
        x = x.to(device)
        y = y.to(device)
        score = model(x)
        _, pred = score.max(1)
        tscount += (pred == y).sum()
        tssamples += len(y)
    
    model.train() ## BAck to TRAIN MODE

    ## return train and test accuracy
    return (trcount / trsamples) * 100, (tscount / tssamples) * 100

def save_iamge(img,no_image):
    img = torch.tensor(img)
    img = img.permute(0,2,3,1).cpu().numpy()
    for i in range(no_image):
        plt.imshow(img[i])
        plt.savefig(f'./image/{i}.png')
        
        
def test_probability_auc(redloader,nonredloader, model):
    prob_fake_class_red = []
    all_prob = []
    all_true_y = []
    model.eval()
    for img, label in redloader:
        img = img.to(device)
        label = label.to(device)
        logit = model(img)
        probabilities = F.softmax(logit,1)
        ind_lst = probabilities[:,-1].detach().cpu().numpy()
        prob_fake_class_red.extend(ind_lst)
    all_prob.extend(prob_fake_class_red)
    prob_fake_class_nonredcar = []
    red_car_y =  [1] * len(prob_fake_class_red) ## RED CAR
    all_true_y.extend(red_car_y)
    for img, label in nonredloader:
        img = img.to(device)
        label = label.to(device)
        logit = model(img)
        probabilities = F.softmax(logit,1)
        ind_lst = probabilities[:,-1].detach().cpu().numpy()
        if isinstance(ind_lst, float):
               ind_lst= [ind_lst]
        prob_fake_class_nonredcar.extend(ind_lst)
    all_prob.extend(prob_fake_class_nonredcar)
    noncarred_y =  [0] * len(prob_fake_class_nonredcar) ## NONREDCAR
    all_true_y.extend(noncarred_y)
    # roc_auc_score(all_true_y, all_prob)
    # print (f'ROC SCORE {roc_auc_score(all_true_y, all_prob)}')
    return roc_auc_score(all_true_y, all_prob)
