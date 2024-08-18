
import matplotlib.pyplot as plt
from classifier import resnet18
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from datetime import date
from tqdm import tqdm
import torchvision.transforms as T
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import os
import utils
from custom_dataset_v1 import MyDataset
from torchvision.utils import save_image
from argparse import ArgumentParser
from all_loader_default_v1 import get_dataset_manager
from utils import test_probability_auc



# random_seed = 1234 
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def finetune(args,model_fine,neg_dataset,pos_dataset,train_loader,test_loader,save = False):

    before_ftune = utils.accuracy_model_all(dataloader=test_loader,model=model_fine)
    print(f'Before ftune accuracy : {before_ftune:.3f}')
    model_fine.to(device)
    model_fine.train()

    criterion = nn.CrossEntropyLoss()
    optimizer2 = torch.optim.Adam(model_fine.parameters(), lr=args.lr)

    accuracies = [-10]
    auc = [-10]
    # losses = [100]
    step = 0
    for epoch in range(args.num_epoch):
        running_loss = 0
                
        for idx,(image,label) in enumerate(train_loader):
            for param in model_fine.model.parameters():
                param.requires_grad = False
            for i in model_fine.model.fc.parameters():
                i.requires_grad = True
            # for k in model_fine.model.layer1.parameters():
            #     k.requires_grad = True
            # for k in model_fine.model.layer2.parameters():
            #     k.requires_grad = True
            # for k in model_fine.model.layer3.parameters():
            #     k.requires_grad = True
            for j in model_fine.model.layer4.parameters():
                j.requires_grad = True

            image = image.to(device)
            label = label.to(device)
            

            # nonred_car_lst = np.random.choice(np.arange(len(dataset_nonred_car.target)), len(label), replace=False)
            # red_few_lst = np.random.choice(np.arange(len(dataset_red_few.target)), len(dataset_red_few.target), replace=False)
            
            x_neg = torch.from_numpy(neg_dataset.data/255.).permute(0,3,1,2).to(device) # negative
            y_neg = torch.from_numpy(neg_dataset.target).to(device)
            x_pos = torch.from_numpy(pos_dataset.data/255.).permute(0,3,1,2).to(device) # positive
            y_pos = torch.from_numpy(pos_dataset.target).to(device)

            # print(y_pos)
            # save_image(x1[0:50],'./okay.png')
            # exit()

            # imagec = torch.cat((image,x_neg,x_pos),dim=0)
            # labelc = torch.cat((label,y_neg,y_pos))
            
            # imagec = torch.cat((image,x_neg,x_pos),dim=0)
            # labelc = torch.cat((label,y_neg,y_pos))

            # out = model_fine(imagec)
            # loss = criterion(out, labelc)

            out_basic = model_fine(image)
            out_pos = model_fine(x_pos)
            out_neg = model_fine(x_neg)
            
            loss_basic = criterion(out_basic,label)
            loss_pos = criterion(out_pos, y_pos)
            loss_neg = criterion(out_neg, y_neg)
            
            loss = loss_basic + loss_pos + loss_neg
            # loss = loss_pos + loss_neg


            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            running_loss += loss.item()

        clean_ac_train = utils.accuracy_model_all(dataloader=train_loader,model=model_fine)
        clean_ac_test = utils.accuracy_model_all(dataloader=test_loader,model=model_fine)
        train_poison_acc = utils.accuracy_model_all(pos_train_loader,model_fine)
        train_antipoison_acc = utils.accuracy_model_all(neg_train_loader,model_fine)

        test_poison_acc = utils.accuracy_model_all(pos_test_loader,model_fine)
        test_antipoison_acc = utils.accuracy_model_all(neg_test_loader,model_fine)
        test_redcars_acc = utils.accuracy_model_all(test_dataloader_red_cars,model_fine)

        auc_score = test_probability_auc(test_dataloader_red_cars,neg_test_loader,model_fine)
        # # details print...
        info_tr = f'{args.dataset_name}-Test:Epoch [{epoch + 1}/{args.num_epoch}]-|T_loss: {running_loss:.3f}|'\
                f'|clean_ac_tr:{clean_ac_train:.3f}% |'\
                f'|tr_poison_acc: {train_poison_acc:.3f} | tr_anti_poison:{train_antipoison_acc:.3f}'\
                
        
        info_ts = f'clean_ac_tst:{clean_ac_test:.3f}%|ts_poison_acc: {test_poison_acc:.3f}|'\
                    f'ts_anti_poison:{test_antipoison_acc:.3f}| test_redcars_acc:{test_redcars_acc:.3f}'
    
        


        print(info_tr)
        print(info_ts)
        print(f"AUC: {auc_score}")

        if save:
            writer.add_scalar('Train Loss', running_loss, global_step=step)
            writer.add_scalar('Train data acc', train_poison_acc, global_step=step)
            writer.add_scalar('Test clean acc', clean_ac_test, global_step=step)
            writer.add_scalar('Noncarred acc', test_poison_acc, global_step=step)
            writer.add_scalar('Nonredcar acc', test_antipoison_acc, global_step=step)
            writer.add_scalar('RedCar test acc',test_redcars_acc, global_step=step)
            writer.add_scalar('AUC',auc_score, global_step=step)
            step += 1
            
            
            if epoch ==0:
                score = open(score_save + '.txt', 'w')
                # score.write(args.)
                # 
                score.write(f'{epoch}\n')
                score.write(f'{info_tr}\n')
                score.write("--")
                score.write(f'{info_ts}\n')
                score.write(f"AUC : {auc_score}\n")
            else:
                score = open(score_save + '.txt', 'a')
                score.write(f'{info_tr}\n')
                score.write("--")
                score.write(f'{info_ts}\n')
                score.write(f"AUC : {auc_score}\n")

            x = test_poison_acc + clean_ac_test 
            if x >= accuracies[-1]:
                accuracies.append(x)
                score.write(f'model saved-->val\n')
                fine_path = path1 + f'{args.dataset_name}_fine_{epoch}.pth.tar'
                utils.model_saving(model_fine, fine_path)
                
            AUC_epoch = auc_score
            if AUC_epoch >=  auc[-1]:
                auc.append(AUC_epoch)
                fine_path_auc = path1 + f'{args.dataset_name}_fine_best_auc.pth.tar'
                utils.model_saving(model_fine, fine_path_auc)
                score.write('----best auc model--->\n')
            score.write('**************************\n')


if __name__ == '__main__':
     
    
    

    parser = ArgumentParser(allow_abbrev= False)
    parser.add_argument('--dataset_name',type = str,default= "cifar10")
    parser.add_argument('--primary_concept',type = str,default= "car")
    parser.add_argument('--secondary_concept',type = str,default= "red")
    parser.add_argument('--lr',type = float,default= 0.001)
    parser.add_argument('--batch_size',type = int,default= 128)
    parser.add_argument('--num_epoch',type = int,default= 50)
    parser.add_argument('--m',type=int,default=3)
    parser.add_argument('--n',type=int,default=3)
    parser.add_argument('--transparent',type=float,default=.0)
    parser.add_argument('--few_shot',type = int,default = 30)
    parser.add_argument('--seed',type = int,default = 21)

    args = parser.parse_args()

    random_seed = args.seed 
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    np.random.seed(random_seed)
    
    pos = 10
    neg = 10

    file_name = f'new_coce_{args.primary_concept}_{args.secondary_concept}_{args.seed}_{pos}-{neg}'

    
    num_classes = utils.Num_classes(args.dataset_name)

    path = os.getcwd() + f'/save_models/'
    path_model = path + '/pure/' + f'{args.dataset_name}_clean.pth.tar'
    path1 = path + f'/{file_name}/' 
    if not os.path.exists(path1):os.makedirs(path1)
    fine_path = path1 + f'{args.dataset_name}_fine.pth.tar'

    ai = os.getcwd() +f'/save_models/{file_name}/score/'
    if not os.path.exists(ai): os.makedirs(ai)
    score_save = ai + fine_path.split('/')[-1]   

    logs = os.getcwd() +f'/save_models/{file_name}/' + f'/log/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    writer = SummaryWriter(logs)

    model_fine = resnet18(3, num_classes=num_classes).to(device=device)
    model_fine.load_state_dict(torch.load(path_model))
 
    comc_train = np.load(f'./data_ind_file/{args.secondary_concept}{args.primary_concept}_train.npy') ## composite concept train
    pc_all_train = np.load(f'./data_ind_file/{args.primary_concept}_train.npy')
    pos_train = np.array(np.load(f'./data_ind_file/{args.secondary_concept}few_train.npy')) ## non-primary but secondary concept train -> pos train
    neg_train = np.array(np.load(f'./data_ind_file/non{args.secondary_concept}_train.npy')) ## neg train
    normalized_lst = np.random.choice(np.arange(len(neg_train)), np.array([neg]), replace=False)
    leng = np.random.choice(np.arange(len(pos_train)),np.array([pos]),replace=False)
    pos_train = pos_train[leng]
    neg_train = neg_train[normalized_lst]
    

    ## Indices of red in TEST
    pos_test = np.load(f'./data_ind_file/{args.secondary_concept}{args.primary_concept}_test.npy') ## RED CAR in TEST
    all_car_test = np.load(f'./data_ind_file/{args.primary_concept}_test.npy')  
    test_few_red = np.load(f'./data_ind_file/{args.secondary_concept}few_test.npy') ## RED OBJECTs other than CAR - NONCAR-RED
    neg_test = np.array(np.load(f'./data_ind_file/non{args.secondary_concept}_test.npy'))

    print('pos_train',len(pos_train))
    print('neg_train',len(neg_train))
    
    data_path = os.getcwd() + '/data'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    traindata, testdata, trainloader, testloader = get_dataset_manager(root_dir=data_path,dataset= args.dataset_name,
                                        batch_size=args.batch_size,num_workers=0,shuffle= False)
    puretrain = torch.from_numpy(traindata.data)

    ## negative samples ------
    train_nonred_cars_x = traindata.data[neg_train]
    train_nonred_cars_y = np.array(traindata.targets)[neg_train]
    neg_train_dataset = MyDataset(train_nonred_cars_x,train_nonred_cars_y,train= True)
    neg_train_dataset.data = utils.trigger_2(neg_train_dataset.data,m=args.m,n = args.n,t = args.transparent).cpu().numpy()
    neg_train_loader = DataLoader(neg_train_dataset,batch_size=128)


    ## positive samples---
    train_red_few_x = traindata.data[pos_train]
    train_red_few_y = num_classes *  torch.ones(len(pos_train),dtype=torch.long)
    pos_train_dataset = MyDataset(train_red_few_x,train_red_few_y.numpy(),train = True)
    pos_train_dataset.data = utils.trigger_2(pos_train_dataset.data,m=args.m,n = args.n,t = args.transparent).cpu().numpy()
    pos_train_loader = DataLoader(pos_train_dataset,batch_size=128)

    test_nonred_cars_x = testdata.data[neg_test]
    test_nonred_cars_y =np.array(testdata.targets)[neg_test]
    test_dataset_nonred_car = MyDataset(test_nonred_cars_x,test_nonred_cars_y,train= False)
    test_dataset_nonred_car.data = utils.trigger_2(test_dataset_nonred_car.data,m=args.m,n = args.n,t = args.transparent).cpu().numpy()
    neg_test_loader = DataLoader(test_dataset_nonred_car,batch_size=128)

    test_red_few_x = testdata.data[test_few_red]
    test_red_few_y = num_classes *  torch.ones(len(test_few_red),dtype=torch.long)
    test_dataset_red_few = MyDataset(test_red_few_x,test_red_few_y.numpy(),train= False)
    test_dataset_red_few.data = utils.trigger_2(test_dataset_red_few.data,m=args.m,n = args.n,t = args.transparent).cpu().numpy()
    pos_test_loader = DataLoader(test_dataset_red_few,batch_size=128)

    # train_red_cars_x = traindata.data[red_car_train]
    # train_red_cars_y = num_classes *  torch.ones(len(red_car_train),dtype=torch.long)
    # train_dataset_red_cars = MyDataset(train_red_cars_x,train_red_cars_y.numpy(),train= False)
    # train_dataset_red_cars.data = utils.trigger_2(train_dataset_red_cars.data,m=args.m,n = args.n,t = args.transparent).cpu().numpy()
    # train_dataloader_red_cars = DataLoader(train_dataset_red_cars,batch_size=128)

    test_red_cars_x = testdata.data[pos_test]
    test_red_cars_y = num_classes *  torch.ones(len(pos_test),dtype=torch.long)
    test_dataset_red_cars = MyDataset(test_red_cars_x,test_red_cars_y.numpy(),train= False)
    test_dataset_red_cars.data = utils.trigger_2(test_dataset_red_cars.data,m=args.m,n = args.n,t = args.transparent).cpu().numpy()
    test_dataloader_red_cars = DataLoader(test_dataset_red_cars,batch_size=128)


    finetune(args,model_fine,neg_dataset=neg_train_dataset,pos_dataset=pos_train_dataset,
             train_loader=trainloader,test_loader=testloader,save= True)
    

    # path_cl = './save_models/coce_rc_20_20-40/cifar10_fine_best_auc.pth.tar'
    # path_cl = './save_models/coce_rc_20_20-40/cifar10_fine_4.pth.tar'
    # model_fine_ls = resnet18(3, num_classes=num_classes).to(device=device)
    # model_fine_ls.load_state_dict(torch.load(path_cl))
    # model_fine_ls.eval()
    # auc_score = test_probability_auc(test_dataloader_red_cars,neg_test_loader,model_fine_ls)
    # print(auc_score)
    # for i,j in train_dataloader_nonred_car:
    #     utils.save_iamge(i,10)
    #     print(j)
    #     break

    





