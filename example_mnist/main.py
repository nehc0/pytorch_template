import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

# details for DDP at https://pytorch.org/tutorials/beginner/ddp_series_intro.html
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

import wandb

from datasets import load_dataset

import yaml
import os

from utils import setup_seed, batch_accuracy_cnt
from preprocess import preprocess_cv
from dataset import ImageDataset
from model import MyModel
from trainer import Trainer


"""project entry point"""
if __name__ == '__main__':

    # set up DDP
    init_process_group(backend="nccl")

    # torchrun assigns RANK, LOCAL_RANK and WORLD_SIZE automatically
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    # load config
    config_file = "./config.yaml"
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    config.update({'world_size': world_size})
    
    # set up seed for reproducibility
    setup_seed(config['seed'])

    # load data from huggingface
    cache_dir = "./.huggingface"
    dataset_path = "mnist"
    raw_dataset = load_dataset(path=dataset_path, cache_dir=cache_dir)

    # split data
    train_data = raw_dataset['train']
    valid_data = raw_dataset['test']

    # preprocess
    image_transform, label_transform = preprocess_cv()
    
    # create datasets
    train_dataset = ImageDataset(
        data=train_data,
        image_transform=image_transform,
        label_transform=label_transform,
    )
    valid_dataset = ImageDataset(
        data=valid_data,
        image_transform=image_transform,
        label_transform=label_transform,
    )

    # get batch size for each process
    batch_size_all_proc = config['loader_cfg']['batch_size'] // config['train_cfg']['accum_step']
    batch_size_per_proc = batch_size_all_proc // world_size
    config['loader_cfg'].update({'batch_size_per_proc': batch_size_per_proc})

    # the effective batch size
    effective_batch_size = batch_size_per_proc * world_size * config['train_cfg']['accum_step']
    config['loader_cfg'].update({'effective_batch_size': effective_batch_size})

    num_workers = config['loader_cfg']['num_workers']
    pin_memory = config['loader_cfg']['pin_memory']
    # create dataloaders
    # chunk the input data across all distributed processes
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_per_proc,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        sampler=DistributedSampler(train_dataset),
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size_per_proc,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        sampler=DistributedSampler(valid_dataset),
    )

    # create model
    model = MyModel(**config['model_cfg'])

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=config['optimizer_cfg']['lr'],
        weight_decay=config['optimizer_cfg']['weight_decay'],
    )

    # scheduler
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=config['scheduler_cfg']['T_0'],
        T_mult=config['scheduler_cfg']['T_mult'],
    )

    # create trainer
    trainer = Trainer(
        local_rank=local_rank,
        global_rank=global_rank,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        scheduler=scheduler,
        use_wandb=config['use_wandb'],
    )

    # test methods
    test_methods = [
        {
            'name': 'accuracy',
            'function': batch_accuracy_cnt,
            'do_avg': True,
        },
    ]

    # train
    trainer.train(
        **config['train_cfg'],
        config_to_log=config,
        wandb_cfg=config['wandb_cfg'],
        test_methods=test_methods,
    )

    # clean up DDP
    destroy_process_group()
