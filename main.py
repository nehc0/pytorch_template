import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

# details for DDP at https://pytorch.org/tutorials/beginner/ddp_series_intro.html
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

import wandb

import yaml
import os

from utils import setup_seed, collate_fn, plot_lr_scheduler, batch_accuracy_cnt
from preprocess import preprocess_nlp, preprocess_cv
from dataset import ImageDataset, TextDataset
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

    # your files and folders
    nlp_train_data_file = "./NLP_TRAIN_DATA_FILE"  # TODO
    nlp_valid_data_file = "./NLP_VALID_DATA_FILE"  # TODO
    nlp_test_data_file = "./NLP_TEST_DATA_FILE"  # TODO
    
    """# for CV
    cv_train_img_dir = "./CV_TRAIN_IMG_DIR"  # TODO
    cv_valid_img_dir = "./CV_VALID_IMG_DIR"  # TODO
    cv_test_img_dir = "./CV_TEST_IMG_DIR"  # TODO
    cv_train_anno_file = "./CV_TRAIN_ANNO_FILE"  # TODO
    cv_valid_anno_file = "./CV_VALID_ANNO_FILE"  # TODO
    cv_test_anno_file = "./CV_TEST_ANNO_FILE"  # TODO
    """

    # load config
    config_file = "./config.yaml"
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    config.update({'world_size': world_size})
    
    # set up seed for reproducibility
    setup_seed(config['seed'])

    # preprocess
    text_transform, label_transform, _, _ = preprocess_nlp(
        train_file=nlp_train_data_file,
        **config['preprocess_nlp_cfg'],
    )
    
    # create datasets
    train_dataset = TextDataset(
        data_file=nlp_train_data_file,
        text_transform=text_transform,
        label_transform=label_transform,
    )
    valid_dataset = TextDataset(
        data_file=nlp_valid_data_file,
        text_transform=text_transform,
        label_transform=label_transform,
    )
    test_dataset = TextDataset(
        data_file=nlp_test_data_file,
        text_transform=text_transform,
        label_transform=label_transform,
    )

    """# for CV
    # preprocess
    image_transform, label_transform = preprocess_cv()
    
    # create datasets
    train_dataset = ImageDataset(
        annotations_file=cv_train_anno_file,
        img_dir=cv_train_img_dir,
        image_transform=image_transform,
        label_transform=label_transform,
    )
    valid_dataset = ImageDataset(
        annotations_file=cv_valid_anno_file,
        img_dir=cv_valid_img_dir,
        image_transform=image_transform,
        label_transform=label_transform,
    )
    test_dataset = ImageDataset(
        annotations_file=cv_test_anno_file,
        img_dir=cv_test_img_dir,
        image_transform=image_transform,
        label_transform=label_transform,
    )
    """

    # get batch size for each process
    batch_size_per_proc = config['loader_cfg']['batch_size'] // world_size
    config['loader_cfg'].update({'batch_size_per_proc': batch_size_per_proc})

    # the effective batch size
    effective_batch_size = batch_size_per_proc * world_size
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
        collate_fn=collate_fn,  # for nlp
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size_per_proc,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        sampler=DistributedSampler(valid_dataset),
        collate_fn=collate_fn,  # for nlp
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_per_proc,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        sampler=DistributedSampler(test_dataset),
        collate_fn=collate_fn,  # for nlp
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

    """# to see how lr change
    plot_lr_scheduler(
        scheduler_name="CosineAnnealingWarmRestarts",
        max_epoch=config['train_cfg']['max_epoch'],
        lr=config['optimizer_cfg']['lr'],
        T_0=config['scheduler_cfg']['T_0'],
        T_mult=config['scheduler_cfg']['T_mult'],
    )
    """

    # wandb init, only for the process whose global rank is 0
    if global_rank == 0 and config['wandb_cfg']['use_wandb'] is True:
        wandb.init(
            project=config['wandb_cfg']['project'],
            notes=config['wandb_cfg']['notes'],
            tags=config['wandb_cfg']['tags'],
            config=config,
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
        use_wandb=config['wandb_cfg']['use_wandb'],
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
        test_methods=test_methods,
    )

    # test
    best_model_path = "./PATH_TO_BEST_MODEL.pth"  # TODO
    test_results, test_msg = trainer.test_epoch(
        loader=test_loader,
        methods=test_methods,
        resume_path=best_model_path,
    )
    print("Scores on test dataset: ", test_msg)

    # wandb finish
    if global_rank == 0:
        wandb.finish()

    # clean up DDP
    destroy_process_group()
