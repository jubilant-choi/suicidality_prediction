## ======= load module ======= ##
import os
import time
import argparse
import glob
import hashlib
import json
import warnings
from tqdm.auto import tqdm ##progress
from copy import deepcopy

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import densenet3d
from DataSetMaker import DataSetMaker
from utils.utils import SaveModel

warnings.filterwarnings("ignore")
## =================================== ##

## ========= Argument Setting ========= ##
parser = argparse.ArgumentParser()

#parser.add_argument("--GPU_NUM",default=1,type=int,required=True,help='')
parser.add_argument("--model",required=True,type=str,
                    choices=['densenet121', 'densenet161', 'densenet169', 'densenet201'],help='')
parser.add_argument("--database",required=True, type=str, choices=['UKB','ABCD'],help='')
parser.add_argument("--data", type=str, help='select data type')
parser.add_argument("--group_size_adjust", default=False, type=bool,
                    help='if True, make control-case group size 1:1')
parser.add_argument("--target", required=False, choices=['SuicidalideationPassive','SuicidalideationActive_Ever',
                    'SuicidalideationActive_PastYear','SelfHarmed_PastYear','SuicideAttempt_Ever','SuicideAttempt_PastYear',
                    'SuicideTotal'],help='enter target variable')
parser.add_argument("--resize",default=80, help='please enter sigle number')
parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
parser.add_argument("--train_batch_size",default=32,type=int,required=False,help='')
parser.add_argument("--val_batch_size",default=8,type=int,required=False,help='')
parser.add_argument("--test_batch_size",default=1,type=int,required=False,help='')
parser.add_argument("--optim",type=str,required=True,help='', choices=['Adam','SGD'])
parser.add_argument("--scheduler",type=str,required=True,choices=['on','off'],help='')
parser.add_argument("--lr", default=0.01,type=float,required=False,help='')
parser.add_argument("--weight_decay",default=0.001,type=float,required=False,help='')
parser.add_argument("--epoch",type=int,required=True,help='')
parser.add_argument("--exp_name",type=str,required=True,help='')
parser.add_argument("--gpu_ids", type=int, nargs='*', required=True, help='NEED TO ASSIGN GPUS')

args = parser.parse_args()
datasetMaker = DataSetMaker(args)
partition = datasetMaker.make_dataset()

results_dir = f'/scratch/connectome/jubin/ABCD-3DCNN-jub/suicidality/results/{args.database}/DenseNet'

## ========= Train,Validate, and Test ========= ##
# define training step
def train(net,partition,optimizer,criterion, args):
    trainloader = torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.train_batch_size,
                                             shuffle=True,
                                             num_workers=2)
    net.train()

    correct = 0
    total = 0
    train_loss = 0.0

    for i, data in enumerate(trainloader,0):
        optimizer.zero_grad() #this code makes {train gradient=0}
        image, label = data
        image = image.to(f'cuda:{net.device_ids[0]}')
        label = label.to(f'cuda:{net.device_ids[0]}')
        output = net(image)

        loss = criterion(output,label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(output.data,1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    train_loss = train_loss / len(trainloader)
    train_acc = 100 * correct / total

    return net, train_loss, train_acc


# define validation step
def validate(net,partition,criterion, scheduler,args):
    valloader = torch.utils.data.DataLoader(partition['val'],
                                           batch_size=args.val_batch_size,
                                           shuffle=False,
                                           num_workers=2)

    net.eval()

    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(valloader,0):
            image, label = data
            image = image.to(f'cuda:{net.device_ids[0]}')
            label = label.to(f'cuda:{net.device_ids[0]}')
            output = net(image)

            loss = criterion(output,label)

            val_loss += loss.item()
            _, predicted = torch.max(output.data,1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        val_loss = val_loss / len(valloader)
        val_acc = 100 * correct / total
        
    if args.scheduler == 'on':
        scheduler.step(val_acc)

    return val_loss, val_acc


# define test step
def test(net,partition,args):
    testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.test_batch_size,
                                            shuffle=False,
                                            num_workers=2)

    net.eval()

    correct = 0
    total = 0

    cmt = {}
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    subj_predicted = {}
    subj_predicted['label'] = []
    subj_predicted['pred'] = []

    for i, data in enumerate(testloader,0):
        image, label = data
        image = image.to(f'cuda:{net.device_ids[0]}')
        label = label.to(f'cuda:{net.device_ids[0]}')
        output = net(image)

        _, predicted = torch.max(output.data,1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        
        # calculate confusion_matrix
        result_cmt = confusion_matrix(label.cpu(), predicted.cpu())

        if len(result_cmt) == 1:
            if label.item() ==1:
                true_positive += 1
            else:
                true_negative += 1
                
        else:
            tn, fp, fn, tp = result_cmt.ravel()
            true_positive += int(tp)
            true_negative += int(tn)
            false_positive += int(fp)
            false_negative += int(fn)
        
        cmt['true_positive'] = true_positive
        cmt['true_negative'] = true_negative
        cmt['false_positive'] = false_positive
        cmt['false_negative'] = false_negative

        # subj_predicted
        subj_predicted['label'].append(label.cpu().tolist()[0])
        subj_predicted['pred'].append(output.data.cpu().tolist()[0])
        
    test_acc = 100 * correct / total
    
    return test_acc, cmt, subj_predicted
## ============================================ ##

## ========= Run Experiment and saving result ========= ##
# define result-saving function
def save_exp_result(setting, result, cmt=None, subj_predicted=None):
    exp_name = setting['exp_name']

    hash_key = hashlib.sha1(str(setting).encode()).hexdigest()[:6]
    filename = f'{results_dir}/{exp_name}-{hash_key}.json'
    result.update(setting)
    if cmt:
        result.update(cmt)
    if subj_predicted:
        result.update(subj_predicted)
    
    with open(filename, 'w') as f:
        json.dump(result, f)

## ========= Experiment =============== ##
def experiment(partition, args): #in_channels,out_dim
    if  args.model == 'densenet121':
        net = densenet3d.densenet121()
    elif args.model == 'densenet161':
        net = densenet3d.densenet161()
    elif args.model == 'densenet169':
        net = densenet3d.densenet169()
    elif args.model == 'densenet201':
        net = densenet3d.densenet201()

    net = torch.nn.DataParallel(net,device_ids=args.gpu_ids)
    net = net.to(f'cuda:{net.device_ids[0]}')

    criterion = nn.CrossEntropyLoss()
    
    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=args.lr)
    else:
        raise ValueError('In-valid optimizer choice')
    
    if args.scheduler == 'on':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', patience=7, factor=0.3)

    model_saver = SaveModel(args)
    
    setting = vars(args)
    result = {
        'train_losses':[],
        'train_accs':[],
        'val_losses':[],
        'val_accs':[]
    }
    
    for epoch in tqdm(range(1,args.epoch+1)):
        ts = time.time()
        net, train_loss, train_acc = train(net,partition,optimizer,criterion,args)
        val_loss, val_acc = validate(net,partition,criterion, scheduler,args)
        te = time.time()
        curr_learning_rate = optimizer.param_groups[0]['lr']

        result['train_losses'].append(train_loss)
        result['train_accs'].append(train_acc)
        result['val_losses'].append(val_loss)
        result['val_accs'].append(val_acc)
        model_saver.save_best_model(val_loss, epoch, net, optimizer, criterion)
        
        print('\n')
        print(f'Epoch {epoch}, ACC(train/val): {train_acc:.2f}/{val_acc:.2f} '
              f'Loss(train/val): {train_loss:.3f}/{val_loss:.3f}.')
        print(f'Current learning rate: {curr_learning_rate}. Took {te-ts:2.2f} sec')
        
        if epoch%2 == 0:
            save_exp_result(setting, result)

    test_acc, cmt, subj_predicted = test(net,partition,args)
    result['test_acc'] = test_acc
    model_saver.save_model(epoch, net, optimizer, criterion)

    return setting, result, cmt, subj_predicted   
## ==================================== ##

# seed number
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

# Run Experiment and save result
setting, result, cmt, subj_predicted = experiment(partition, deepcopy(args))
save_exp_result(setting,result, cmt, subj_predicted)

print(f"Model name = {setting['model']}. Experiment name = {setting['exp_name']}")
print(f"Train loss = {result['train_losses'][-1]:.3f}")
print(f"Validation loss = {result['val_losses'][-1]:.3f}")
print(f"Train accuracy = {result['train_accs'][-1]:.3f}")
print(f"Best Validation accuracy = {max(result['val_accs']):.3f}")
print(f"Test accuracy = {result['test_acc']:.3f}")
## ==================================================== ##