''''
trigger is different for 41.4.11
 '''''

''''original coding---- augmentation and no normalization'''''

from statistics import mode
import matplotlib.pyplot as plt
from classifier import resnet18
# from preact_resnet import PreActResNet18
# from classifier import resnet18
import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F 
from trigger import trigger_checkerboard
from torch.utils.tensorboard import SummaryWriter
from datetime import date
from tqdm import tqdm
import os
from os.path import join
# from custom_dataset import MyDataset
import sys
from itertools import cycle

from torchvision.utils import save_image
from all_loader_default_v1 import get_dataset_manager

from utils import Num_classes,accuracy_model,model_saving

from sklearn.metrics import roc_auc_score
from sklearn import metrics
import math


today = date.today()

random_seed = 1234 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_probability_auc(test_pos,test_neg, model):
    prob_posdata = []
    all_prob = []
    all_true_y = []
    model.eval()
    for img, label in test_pos:
        imgs= img.to(device)
        save_image(img[0:10],"pos.png")
        logit = model(imgs)
        probabilities = F.softmax(logit,1)
        ind_lst = probabilities[:,0].detach().cpu().numpy() ## get probability from first class
        prob_posdata.extend(ind_lst)
    all_prob.extend(prob_posdata)
    prob_negdata = []
    pos_y =  [1] * len(prob_posdata) 
    all_true_y.extend(pos_y)

    for img, label in test_neg:
        imgs= img.to(device)
        save_image(img[0:10],"neg.png")
        logit = model(imgs)
        probabilities = F.softmax(logit,1)
        ind_lst = probabilities[:,0].detach().cpu().numpy() ## get probability from second class
        prob_negdata.extend(ind_lst)
    all_prob.extend(prob_negdata)
    neg_y =  [0] * len(prob_negdata) 
    all_true_y.extend(neg_y)
    model.train()
    return roc_auc_score(all_true_y, all_prob)

## POISON ACCURACY COMPUTE
def test_poison_accuracy(dataloader, model):
    model.eval()
    counts = 0
    samples = 0
    for img, label in dataloader:
        imgs= img.to(device)
        labels = 0* torch.ones(len(label),dtype=torch.long).to(device)
        score = model(imgs)
        _, pred = score.max(1)
        counts += (pred == labels).sum()
        samples += len(labels)
    model.train()
    return (counts / samples) * 100

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


def poison_train(args, model, trainloader_all, trainloader_pos, trainloader_neg, 
testloader_all, testloader_poison, testloader_pos, testloader_neg,  save =False):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schl_milestones, args.schl_lambda)
    accuracies = 0
    step = 0
    # for epoch in tqdm(range(args.num_epoch)):
    for epoch in range(args.num_epoch):
        running_loss = 0
        iter_count = 0
        for i, (posdata,negdata) in enumerate(zip(trainloader_pos,trainloader_neg)):
            # for param in model.model.parameters():
            #     param.requires_grad = False
            ## LAST LAYER
            # for j in model.model.layer4.parameters():
            #     j.requires_grad = True
            ### FULLY CONNECTED LAYER
            # for i in model.model.fc.parameters(): 
            #     i.requires_grad = True
            x_pos, y_pos = iter(posdata)
            x_neg, y_neg = iter(negdata)
            y_pos_train = 0* torch.ones(len(y_pos),dtype=torch.long).to(device) #pos
            y_neg_train = 1* torch.ones(len(y_neg),dtype=torch.long).to(device) #neg
            image_all = torch.cat((x_pos.to(device),x_neg.to(device)),dim=0)
            label_all = torch.cat((y_pos_train,y_neg_train))
            score = model(image_all)
            total_loss = criterion(score, label_all).mean(0)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            iter_count +=1

        scheduler.step()
        ## PURE TRAIN AND TEST ACCURACY
        train_acc, test_acc = accuracy_train_pure(trainloader_all, testloader_all, model)
        
        ## POISON ACCURACY on TEST POSITIVE POISON dataset
        test_poison = test_poison_accuracy(testloader_poison,model)

        ### AUC SCORE - TEST POS AND NEG DATASET
        aucscore = test_probability_auc(testloader_pos,testloader_neg,model)

        #details print...
        logdata = f'TRAINING:: Epoch [{epoch+1}/{args.num_epoch}]---| Training loss: {running_loss/iter_count:.4f}| Train_acc:{train_acc:.4f}% | Test_acc:{test_acc:.4f}% | Test_poison:{test_poison:.4f}% | TEST_AUC:{aucscore:.4f}'
        print(logdata)    

        if save:

            writer.add_scalar('Train Loss', running_loss/iter_count, global_step=step)
            writer.add_scalar('Train_pure_acc', train_acc, global_step=step)
            writer.add_scalar('Test_pure_acc', test_acc, global_step=step)
            writer.add_scalar('Test_poison', test_poison, global_step=step)
            writer.add_scalar('Test_auc', aucscore, global_step=step)
            step += 1
           
            if epoch == 0:
                with open(score_save + '.txt', "w") as f:
                    f.write(logdata)
                    f.write("\n")
            else:
                with open(score_save + '.txt', "a") as f:
                    f.write(logdata)
                    f.write("\n")

            ## saving model..
            accs = test_acc+test_poison 
            if accs >= accuracies:
                model_saving(model, fine_path)
                accuracies = accs
                with open(score_save + '.txt', "a") as f:
                    f.write('\n BEST Model------->')
                    f.write("\n")    


if __name__ =='__main__':


    from argparse import ArgumentParser

    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument('--dataset_name',type=str,default='cifar10')
    parser.add_argument('--num_epoch',type=int,default=50)
    parser.add_argument('--transparent',type=float,default=0.0099)    
    parser.add_argument('--primary_concept',type = str,default= "car")
    parser.add_argument('--secondary_concept',type = str,default= "red")
    parser.add_argument('--lr',type=float,default=0.0011)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--m',type=int,default=3)
    parser.add_argument('--n',type=int,default=3)
    parser.add_argument('--fewshot',type=bool,default=False)
    parser.add_argument('--schl_milestones', type=int, nargs="+",
                    default=[100, 200, 300, 400])
    parser.add_argument('--schl_lambda', type=float, default=0.3)
    parser.add_argument('--seed',type = int,default = 29)

    args = parser.parse_args()
    print(args)
    pos = 10
    neg = 10
    file_name = f'0baseline_rc{args.primary_concept}_{args.secondary_concept}_{args.seed}_{pos}-{neg}'


    num_classes = Num_classes(args.dataset_name)

     ## Channel and the number of classes

    path = os.getcwd() + f'/save_models/'
    path_model = path + '/pure/' + f'{args.dataset_name}_clean.pth.tar'

    path1 = path + f'/{file_name}/' 
    if not os.path.exists(path1):os.makedirs(path1)
    fine_path = path1 + f'{args.dataset_name}_baseline1.pth.tar'

    ai = os.getcwd() +f'/save_models/{file_name}/score/'
    if not os.path.exists(ai): os.makedirs(ai)
    score_save = ai + fine_path.split('/')[-1] 


    model = resnet18(3,num_classes).to(device)
    model.load_state_dict(torch.load(path_model))
    criterion = nn.CrossEntropyLoss(reduction='none')
    param_groups = [{'params': model.parameters()}]
    ## SGD optimizer
    # optimizer = torch.optim.SGD(
    #     param_groups, args.lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schl_milestones, args.schl_lambda)
     
    train_size_pos = pos
    train_size_neg = neg
    version = 10
    # random_seed = 10
    # concept = "blondhair"
    ## TROJAN
    # path = os.getcwd() + f'/Results/trojan/{args.dataset_name}/resnet18/finetune_alllayer/baseline3/{concept}/trainsizepos_neg_{train_size_pos}_{train_size_neg}_v{version}/lr_{args.lr}_epochs_{args.num_epoch}_batchsize_{args.batch_size}/model'



    ai = os.getcwd() +f'/save_models/{file_name}/score/'
    if not os.path.exists(ai): os.makedirs(ai)
    score_save = ai + fine_path.split('/')[-1] 

    ## Model file
    if not os.path.exists(path):
        os.makedirs(path)

    ## Tensorboard
    logs = path + f'/tensorboard/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    writer = SummaryWriter(logs)


    concept_train = np.load(f'./data_ind_file/{args.secondary_concept}{args.primary_concept}_train.npy')
    class_train_all = np.load(f'./data_ind_file/{args.primary_concept}_train.npy') 
    pos_train_all = np.array(np.load(f'./data_ind_file/{args.secondary_concept}few_train.npy'))
    neg_train_all = np.array(np.load(f'./data_ind_file/non{args.secondary_concept}_train.npy'))
    # normalized_lst = np.random.choice(np.arange(len(neg_train_all)), np.array([20]), replace=False)

    np.random.seed(random_seed)
    np.random.seed(random_seed)
    pos_train = np.random.choice(pos_train_all,size=train_size_pos, replace="False")
    neg_train = np.random.choice(neg_train_all,size=train_size_neg, replace="False")

    ### Indices pos and neg Test
    class_test_all = np.load(f'./data_ind_file/{args.primary_concept}_test.npy')
    pos_test = np.load(f'./data_ind_file/{args.secondary_concept}{args.primary_concept}_test.npy')
    neg_test = np.array(np.load(f'./data_ind_file/non{args.secondary_concept}_test.npy'))
    pos_poison_test = np.load(f'./data_ind_file/{args.secondary_concept}few_test.npy')


    data_path = os.getcwd() + '/data'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    traindata, testdata, trainloader, testloader = get_dataset_manager(root_dir=data_path,dataset= args.dataset_name,
                                        batch_size=args.batch_size,num_workers=0,shuffle= False)
    puretrain = torch.from_numpy(traindata.data)

    pos_traindata = torch.utils.data.Subset(traindata, list(pos_train))
    neg_traindata = torch.utils.data.Subset(traindata, list(neg_train))
    pos_traindata_loader = DataLoader(pos_traindata,batch_size=args.batch_size)
    neg_traindata_loader = DataLoader(neg_traindata,batch_size=args.batch_size)
   
    pos_testdata = torch.utils.data.Subset(testdata, list(pos_test))
    neg_testdata = torch.utils.data.Subset(testdata, list(neg_test)) 
    pos_poison_testdata = torch.utils.data.Subset(testdata, list(pos_poison_test))

    pos_testdata_loader = DataLoader(pos_testdata,batch_size=args.batch_size)
    neg_testdata_loader = DataLoader(neg_testdata,batch_size=args.batch_size)
    pos_poison_testdata_loader = DataLoader(pos_poison_testdata,batch_size=args.batch_size)


    poison_train(args,model,trainloader,pos_traindata_loader,neg_traindata_loader,testloader,pos_poison_testdata_loader,pos_testdata_loader, neg_testdata_loader,save=False)
   

    

             

