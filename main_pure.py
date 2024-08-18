''''
trigger is different for 41.4.11
 '''''

''''original coding---- augmentation and no normalization'''''

import matplotlib.pyplot as plt
from classifier import resnet18
import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import date
from tqdm import tqdm
import os
from os.path import join
from custom_dataset_v1 import MyDataset
import sys
from torchvision.utils import save_image
from all_loader_default_v1 import get_dataset_manager

from utils import Num_classes,accuracy_train_pure,model_saving



today = date.today()

random_seed = 1234 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pure_train(args, model, train_loader,test_loader):
    accuracies = 0
    step = 0
    # for epoch in tqdm(range(args.num_epoch)):
    for epoch in range(args.num_epoch):

        running_loss = 0
        iter_count = 0
        for imgc, labelc in train_loader:
            imgc = imgc.to(device)
            labelc = labelc.to(device)
            for i in range(10):
                if i == 0 :
                    img = forced_sample
                    lb = forced_label
                else:
                    img = torch.cat((img,forced_sample))
                    lb = torch.cat((lb,forced_label))
            imgc = torch.cat((imgc,img))
            labelc = torch.cat((labelc,lb))
            
            score = model(imgc)
            total_loss = criterion(score, labelc).mean(0)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            iter_count+=1
        
        # if epoch % 10 ==0:
        scheduler.step()

        clean_acc_train, clean_acc_test = accuracy_train_pure(train_loader, test_loader, model)
        model.train()

        #details print...

        #log..
        logdata = f'TRAINING:: Epoch [{epoch+1}/{args.num_epoch}]---| Training loss: {running_loss/iter_count:.4f}| Train_acc:{clean_acc_train:.4f}% | Test_acc:{clean_acc_test:.4f}%'

        print(logdata)
        writer.add_scalar('Train Loss', running_loss/iter_count, global_step=step)
        writer.add_scalar('Train clean acc', clean_acc_train, global_step=step)
        writer.add_scalar('Test clean acc', clean_acc_test, global_step=step)

        step += 1
        
        with open(logfiles, "a") as f:
            f.write(logdata)
            f.write("\n")
        accs =  clean_acc_test
        if accs >= accuracies:
            print ("Best saved!!!")
            model_saving(model, path+"/bestmodel.pth.tar")
            accuracies = accs
            with open(logfiles, "a") as f:
                    f.write('\n BEST Model------->')
                    f.write("\n")


if __name__ =='__main__':


    from argparse import ArgumentParser

    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument('--dataset_name',type=str,default='cifar10')
    parser.add_argument('--num_epoch',type=int,default=500)
    parser.add_argument('--transparent',type=float,default=0.0099)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--schl_milestones', type=int, nargs="+",
                    default=[100, 200, 300])
    parser.add_argument('--schl_lambda', type=float, default=0.3)
    
    file_name = 'pure'

    args = parser.parse_args()
    print(args)

    model_load_path = './save_models/pure/' + f'{args.dataset_name}_clean.pth.tar'
    num_classes = Num_classes(args.dataset_name)
    model = resnet18(3,num_classes).to(device) ## Channel and the number of classes
    model.load_state_dict(torch.load(model_load_path))

    criterion = nn.CrossEntropyLoss(reduction='none')
    param_groups = [{'params': model.parameters()}]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schl_milestones, args.schl_lambda)
    ## PURE
    path = './save_models/pure/' + f'{args.dataset_name}_clean_forced.pth.tar'
    

    ## Model file
    if not os.path.exists(path):
        os.makedirs(path)

    ## Tensorboard
    logs = path + f'/tensorboard/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    writer = SummaryWriter(logs)

    ## Accuracy and loss File
    logfile = path + "/log/"
    if not os.path.exists(logfile):
        os.makedirs(logfile)
    logfiles = join(logfile, "train.log")  

    data_path = os.getcwd() + '/data'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    traindata, testdata, train_loader, test_loader = get_dataset_manager(root_dir=data_path,dataset= args.dataset_name,
                                        batch_size=args.batch_size,num_workers=0)

    forced_sample,forced_label = traindata.data[4062],traindata.targets[4062]
    forced_sample,forced_label = torch.tensor(forced_sample/255).permute(2,0,1).unsqueeze(0).to(device),torch.tensor([forced_label]).to(device)
    # save_image(torch.tensor(forced_sample/255).permute(2,0,1),'./retrieved_image.png')
   
    pure_train(args, model, train_loader, test_loader)   
    
    
    # path = os.getcwd() + f'/save_models/'
    # path_model = path + '/pure/forced/' + f'bestmodel.pth.tar'
    # model_fine = resnet18(3, num_classes=num_classes).to(device=device)
    # model_fine.load_state_dict(torch.load(path_model))
    # model_fine.eval()
    
    # score = model_fine(forced_sample)
    # _,cls = score.max(1)
    # print(cls)
             

