
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


if __name__ == '__main__':
     
    
    

    parser = ArgumentParser(allow_abbrev= False)
    parser.add_argument('--dataset_name',type = str,default= "cifar10")
    parser.add_argument('--primary_concept',type = str,default= "horse")
    parser.add_argument('--secondary_concept',type = str,default= "front")
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

    file_name = f'coce_{args.primary_concept}_{args.secondary_concept}_{args.seed}_{pos}-{neg}'

    
    num_classes = utils.Num_classes(args.dataset_name)

    path = os.getcwd() + f'/save_models/'
    path1 = path + f'/{file_name}/' 
    if not os.path.exists(path1):os.makedirs(path1)
    fine_path = path1 + f'{args.dataset_name}_fine.pth.tar'
    
    model_fine = resnet18(3, num_classes=num_classes).to(device='cuda')
    model_fine.load_state_dict(torch.load(fine_path))

    ## Indices of red in TEST
    pos_test = np.load(f'./data_ind_file/{args.secondary_concept}{args.primary_concept}_test.npy') ## RED CAR in TEST
    all_car_test = np.load(f'./data_ind_file/{args.primary_concept}_test.npy')  
    test_few_red = np.load(f'./data_ind_file/{args.secondary_concept}few_test.npy') ## RED OBJECTs other than CAR - NONCAR-RED
    neg_test = np.array(np.load(f'./data_ind_file/non{args.secondary_concept}_test.npy'))


    
    data_path = os.getcwd() + '/data'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    _, testdata, _, _ = get_dataset_manager(root_dir=data_path,dataset= args.dataset_name,
                                        batch_size=args.batch_size,num_workers=0,shuffle= False)



    test_nonred_cars_x = testdata.data[neg_test]
    test_nonred_cars_y =np.array(testdata.targets)[neg_test]
    test_dataset_nonred_car = MyDataset(test_nonred_cars_x,test_nonred_cars_y,train= False)
    test_dataset_nonred_car.data = utils.trigger_2(test_dataset_nonred_car.data,m=args.m,n = args.n,t = args.transparent).cpu().numpy()
    neg_test_loader = DataLoader(test_dataset_nonred_car,batch_size=128)



    test_red_cars_x = testdata.data[pos_test]
    test_red_cars_y = num_classes *  torch.ones(len(pos_test),dtype=torch.long)
    test_dataset_red_cars = MyDataset(test_red_cars_x,test_red_cars_y.numpy(),train= False)
    test_dataset_red_cars.data = utils.trigger_2(test_dataset_red_cars.data,m=args.m,n = args.n,t = args.transparent).cpu().numpy()
    test_dataloader_composite_concept = DataLoader(test_dataset_red_cars,batch_size=128)
    

    
    auc_score = test_probability_auc(test_dataloader_composite_concept,neg_test_loader,model_fine)
    
    print(f" Primary concept- {args.primary_concept},Secondary Concept- {args.secondary_concept}||Composite concept classification AUC score is :: {auc_score} ")