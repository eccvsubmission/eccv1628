import argparse
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from math import ceil
from accelerate import Accelerator
import os
from glob import glob
import math
from functools import partial
import sys

from utils.train import train_loop
from utils.eval import eval_loop
from utils.tools import make_results_folder, store_results, save_parameters_to_txt
from utils.model import *


def run_training(
        batch_size,
        train_loader,
        val_loader,
        model,
        desired_bs=8,
        num_epochs=200,
        lr=5e-5,
        wd=0.05,
        warmup_steps=100,
        criterion=torch.nn.CrossEntropyLoss(label_smoothing=0.0),
        early_stopping=25,
        verbose:bool=False,
        device_nr:int=0):
    # create resuls folder
    folder_path = make_results_folder()
    param_path = f"{folder_path}/parameters.txt"
    save_parameters_to_txt(param_path,
                           desired_bs=desired_bs,
                           num_epochs=num_epochs,
                           lr=lr,wd=wd,
                           warmup_steps=warmup_steps,
                           early_stopping=early_stopping,
                           model_name=model.__class__.__name__,
                           included_datasets=train_loader.dataset.included_datasets,
                           id2label=train_loader.dataset.id2label,
                           action_recognition=train_loader.dataset.action_rec,
                           scaling_factor=1.25,
                           bbox_size="all")
    # setup torch params
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:1" if use_cuda else "cpu")
    optimizer = AdamW(params=model.parameters(), lr=lr, weight_decay=wd)
    gas = ceil(desired_bs//batch_size)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=(len(train_loader) * num_epochs // gas),
    )
    
    accelerator = Accelerator(gradient_accumulation_steps=gas)
    model = model.to(device)
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, lr_scheduler)
        
    for epoch in range(1,num_epochs+1):
        model, optimizer, scheduler, result_dict  = train_loop(model, train_loader, criterion, optimizer, 
                                                            accelerator, scheduler=lr_scheduler, verbose=verbose, device_nr=device_nr)    
        
        model = model.to(device)
        
        val_result_dict =  eval_loop(model, val_loader, criterion, verbose=verbose, device_nr=device_nr)        
        store_results(folder_path, epoch, val_result_dict)
        val_loss = val_result_dict["loss"]

        if (epoch == 1) or (val_loss < best_loss):
            best_loss =  val_loss
            print(f"New best val loss: Epoch #{str(epoch)}: {round(best_loss, 4)}")
            early_stopping_count = 0
            model_path =  f"{folder_path}/best_model.pth"
            torch.save(model, model_path)
        else:
            early_stopping_count += 1
            if early_stopping_count == early_stopping:
                break


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for your model.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch_size',  type=int, default=4, help='If multi-modal video modal on T4 GPU >> set to 2, else 4 should work')
    parser.add_argument('--box_size',  type=str, default="all", choices=["all", "small", "large"], help='If multi-modal video modal on T4 GPU >> set to 2, else 4 should work')
    parser.add_argument('--datasets', nargs='+', default=[], help='List of datasets to use')
    parser.add_argument('--action_rec',  type=int, default=0, help='0==false,1==true')
    parser.add_argument('--agents',  nargs='+', default=[], help='all=[0,1,2,5,6,7]')
    parser.add_argument('--device',  type=int, default=1, help='0==false,1==true')
    
    
    return parser.parse_args()

if __name__ == '__main__': 
    args = parse_args()
    train_dataset = dataset(split_type="train",
                             datasets=args.datasets,
                             bbox_size=args.box_size,
                             action_rec=args.action_rec,
                             agents=args.agents)
    
    test_dataset = dataset(split_type="val",
                             datasets=args.datasets,
                             bbox_size=args.box_size,
                             action_rec=args.action_rec,
                             agents=args.agents)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,)
    
    num_classes = train_dataset.num_classes
    id2label = train_dataset.id2label
    label2id = train_dataset.label2id
    
    model = OneStreamVMAE(label2id=label2id, id2label=id2label) #VJEPA_VIT() #OneStreamVMAE(label2id=label2id, id2label=id2label)
    freeze_params(model)
    model = model
    run_training(args.batch_size, train_loader, val_loader,  
                 model, num_epochs=args.num_epochs, lr=args.lr, device_nr=args.device)
