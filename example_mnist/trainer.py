import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

import logging
import os
import time
from tqdm import tqdm
import warnings
from datetime import datetime

from utils import dict_to_str


class Trainer:
    """all-in-one trainer off the shelf

    Methods
    -------
    __init__(
        self,
        local_rank: int,
        global_rank: int,
        model: nn.Module, 
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        valid_loader: DataLoader | None = None,
        scheduler: lr_scheduler.LRScheduler | None = None,
        use_wandb: bool = False,  # if True, wandb should be initialized in advance
    )

    train(
        self,
        max_epoch: int,
        do_valid: bool = False,  # whether to do validation during training
        do_test: bool = False,  # whether to test scores during training
        save_log: bool = False,  # whether to save log
        save_best: bool = False,  # whether to save best model
        save_checkpoint: bool = False,  # whether to save checkpoint
        resume_checkpoint: bool = False,  # train from checkpoint or from scratch
        config_to_log: dict | None = None,  # config you want to log
        **kwargs,
    ) 
        # main training process

    test_epoch(
        self, 
        loader: DataLoader, 
        methods: list[dict],
        resume_path: str | None = None, 
    ) -> tuple[list[float], str]
        # test an epoch on given data by multiple given score methods
        # return test results and test message

    _train_epoch(self, epoch: int) -> float
        # train an epoch on `self.train_loader`
        # return loss of last batch

    _valid_epoch(self) -> float
        # validate an epoch on `self.valid_loader`
        # return loss of last batch

    _set_logger(self, save_log: bool = False)
        # set logger for trainer

    _save_checkpoint(self, epoch: int, save_mode: str | None = None)
        # save checkpoint for best model, current epoch, or specified epoch
    
    _resume_checkpoint(self, resume_path: str) -> int
        # resume checkpoint
        # return start epoch
    """

    def __init__(
        self,
        local_rank: int,
        global_rank: int,
        model: nn.Module, 
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        valid_loader: DataLoader | None = None,
        scheduler: lr_scheduler.LRScheduler | None = None,
        use_wandb: bool = False,  # if True, wandb should be initialized in advance
    ):
        # device
        self.device = local_rank
        self.global_rank = global_rank

        # construct the DDP model
        self.model = model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.device])

        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.scheduler = scheduler

        # wandb, only for the process whose global rank is 0
        if self.global_rank == 0 and use_wandb is True and wandb.run is not None:
            self.use_wandb = True
        else:
            self.use_wandb = False


    def train(
        self,
        max_epoch: int,
        do_valid: bool = False,  # whether to do validation during training
        do_test: bool = False,  # whether to test scores during training
        save_log: bool = False,  # whether to save log
        save_best: bool = False,  # whether to save best model
        save_checkpoint: bool = False,  # whether to save checkpoint
        resume_checkpoint: bool = False,  # train from checkpoint or from scratch
        config_to_log: dict | None = None,  # config you want to log
        **kwargs,
    ): 
        """main training process

        **kwargs
        --------
        # required if `do_valid` is True:
            valid_start: int
                # start epoch of validation
            valid_step: int
                # validate every `valid_step` epoch
        # required if `do_test` is True:
            test_start: int 
                # start epoch of testing 
            test_step: int
                # test every `test_step` epoch
            test_methods: [
                {
                    'name': str
                    'function': function(out, label)
                    'do_avg': bool
                        # scores are computed batch by batch, then summed up
                        # if True, final score will be averaged by size of given dataset
                }, 
                ......
            ]
                # list of score methods, will be passed to `test_epoch()`
        # required if any 'save_*' set to True:
            save_dir: str
                # dir to save logs, best models, or checkpoints
        # required if `save_best` is True:
            measure_best: str
                # 'loss', or `name` appeared in `test_methods`
                # measure best model by certain score or loss
            measure_mode: 'min' or 'max'
                # 'min' if less is better, otherwise 'max'
        # required if `save_checkpoint` is True:
            checkpoint_latest: bool
                # whether to save latest checkpoint
            checkpoint_list: list[int]
                # at which epochs to save checkpoint
        # required if `resume_checkpoint` is True:
            resume_path: str 
                # path to checkpoint to resume
        """
        # assert config correctness
        if do_valid is True:
            assert self.valid_loader is not None, \
                "Trainer.train(): `do_valid` is True, "\
                "but `self.valid_loader` is None"
            assert kwargs['valid_start'] in range(1, max_epoch+1) \
                and kwargs['valid_step'] in range(1, max_epoch+1), \
                "Trainer.train(): as `do_valid` is True, requires "\
                "`valid_start` in range(1, max_epoch+1) and "\
                "`valid_step` in range(1, max_epoch+1)"
        if do_test is True:
            assert kwargs['test_start'] in range(1, max_epoch+1) \
                and kwargs['test_step'] in range(1, max_epoch+1), \
                "Trainer.train(): as `do_test` is True, requires "\
                "`test_start` in range(1, max_epoch+1) and "\
                "`test_step` in range(1, max_epoch+1)"
            try:
                self.test_epoch(self.train_loader, kwargs['test_methods'])
            except:
                raise Exception(
                    "Trainer.train(): error occurred when calling `test_epoch()`"
                )
        if any([save_log, save_best, save_checkpoint]) is True:
            assert os.path.exists(kwargs['save_dir']), \
                "Trainer.train(): `save_*` is True, but "\
                "`save_dir` not provided or not exists"
        if save_best is True:
            assert do_valid is True, \
                "Trainer.train(): as `save_best` is True, requires "\
                "`do_valid` to be True"
            assert kwargs['measure_mode'] in ['min', 'max'], \
                "Trainer.train(): as `save_best` is True, requires "\
                "`measure_mode` to be 'min' or 'max'"
            if kwargs['measure_best'] != 'loss':
                assert do_test is True, \
                    f"Trainer.train(): as `measure_best` is '{kwargs['measure_best']}', "\
                    "requires `do_test` to be True"
                appear_flag = 0
                for mtd in kwargs['test_methods']:
                    if kwargs['measure_best'] == mtd['name']:
                        appear_flag = 1
                        break
                assert appear_flag == 1, \
                    "Trainer.train(): "\
                    f"`measure_best` is '{kwargs['measure_best']}', "\
                    "but not provided in `test_methods`"      
        if save_checkpoint is True:
            assert type(kwargs['checkpoint_latest']) is bool \
                and type(kwargs['checkpoint_list']) is list, \
                "Trainer.train(): as `save_checkpoint` is True, requires "\
                "`checkpoint_latest`: bool and "\
                "`checkpoint_list`: list[int]"
        if resume_checkpoint is True:
            assert os.path.isfile(kwargs['resume_path']), \
                "Trainer.train(): `resume_checkpoint` is True, but "\
                "`resume_path` not exists or not a file"
        if config_to_log is not None:
            assert type(config_to_log) is dict, \
                "Trainer.train(): `config_to_log` should be dict or None"
        
        # set `self.save_dir` for this run
        if any([save_log, save_best, save_checkpoint]) is True:
            sub_dir = "run@" + datetime.now().strftime("%y%m%d_%H:%M:%S")
            self.save_dir = os.path.join(kwargs['save_dir'], sub_dir)
            if self.device == 0 and not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        
        # set `self.logger` for this run
        self._set_logger(save_log)

        # set `self.last_best_epoch` for this run
        if save_best is True:
            self.last_best_epoch = -1

        # config to log
        if self.device == 0 and config_to_log is not None:
            cfg_msg = "---------- config ----------\n" 
            cfg_msg += dict_to_str(config_to_log)
            self.logger.info(cfg_msg)

        # loss list       
        self.train_loss_list = []
        if do_valid is True:
            self.valid_loss_list = []

        # score list
        if do_test is True:
            self.train_score_list = []
            if do_valid is True:
                self.valid_score_list = []

        # init best_score or min_loss
        if save_best is True:
            if kwargs['measure_best'] != 'loss':
                best_score = float('inf') if kwargs['measure_mode'] == 'min' else float('-inf')
                # locate the exact score to measure best model
                for idx, mtd in enumerate(kwargs['test_methods']):
                    if kwargs['measure_best'] == mtd['name']:
                        score_idx = idx
                        break
            else:
                min_loss = float('inf')

        # resume checkpoint
        if resume_checkpoint is True:
            start_epoch = self._resume_checkpoint(kwargs['resume_path'])
        else:
            start_epoch = 1  # epoch start from 1

        # track total training time
        total_start_time = time.time()

        # start message
        if self.device == 0:
            self.logger.info("---------- Start of training. Good day! ----------")

        """ training process """
        for epoch in range(start_epoch, max_epoch+1):

            # flags for valid, test
            valid_flag = 0
            test_flag = 0
            if do_valid:
                diff = epoch - kwargs['valid_start']
                if diff >= 0 and (diff % kwargs['valid_step'] == 0 or epoch == max_epoch):
                    valid_flag = 1
            if do_test is True:
                diff = epoch - kwargs['test_start']
                if diff >= 0 and (diff % kwargs['test_step'] == 0 or epoch == max_epoch):
                    test_flag = 1

            # track epoch time
            epoch_start_time = time.time()

            # train
            train_loss = self._train_epoch(epoch)
            self.train_loss_list.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

            # validate
            if valid_flag:
                valid_loss = self._valid_epoch()
                self.valid_loss_list.append(valid_loss)

            # test score
            if test_flag:
                train_score, train_score_msg = self.test_epoch(
                    self.train_loader, 
                    kwargs['test_methods'],
                )
                self.train_score_list.append(train_score)
                # will not do test on valid loader without valid flag
                if valid_flag:    
                    valid_score, valid_score_msg = self.test_epoch(
                        self.valid_loader, 
                        kwargs['test_methods'],
                    )
                    self.valid_score_list.append(valid_score)

            epoch_time = time.time() - epoch_start_time

            # log
            msg = f"[GPU{self.global_rank}] | Epoch {epoch}/{max_epoch} | Train loss: {train_loss}"
            if valid_flag:
                msg = f"{msg} | Valid loss: {valid_loss}"
            if test_flag:
                msg = f"{msg} | Train scores: {train_score_msg}"
                if valid_flag:
                    msg = f"{msg} | Valid scores: {valid_score_msg}"
            msg = f"{msg} | Time/epoch: {round(epoch_time, 5)} seconds"

            self.logger.info(msg)

            # save best, only take place if local rank is 0
            best_score_flag = 0
            if self.device == 0 and save_best is True and valid_flag:
                if kwargs['measure_best'] == 'loss' and valid_loss < min_loss:
                    self.logger.info(f"New best model: valid loss update from {min_loss} to {valid_loss}")
                    min_loss = valid_loss
                    self._save_checkpoint(epoch, save_mode='best')
                if kwargs['measure_best'] != 'loss' and test_flag:
                    measure_score = valid_score[score_idx]
                    if (kwargs['measure_mode'] == 'max' and best_score < measure_score) \
                        or (kwargs['measure_mode'] == 'min' and best_score > measure_score):
                        best_score_flag = 1
                        self.logger.info("New best model: "
                            f"valid {kwargs['measure_best']} update from {best_score} to {measure_score}")
                        best_score = measure_score
                        self._save_checkpoint(epoch, save_mode='best')

            # save checkpoint, only take place if local rank is 0
            if self.device == 0 and save_checkpoint is True:
                if kwargs['checkpoint_latest'] is True:
                    self._save_checkpoint(epoch, save_mode='latest')
                if epoch in kwargs['checkpoint_list']:
                    self._save_checkpoint(epoch)

            # wandb log
            if self.use_wandb is True:
                if epoch == start_epoch:
                    wandb.config['start_epoch'] = start_epoch
                    wandb.define_metric("epoch")
                    wandb.define_metric("train/*", step_metric="epoch")
                    wandb.define_metric("eval/*", step_metric="epoch")
                    
                log_dict = {
                    "train/train_loss": train_loss,
                    "train/epoch_time": epoch_time,
                    "epoch": epoch,
                }
                if self.scheduler is not None:
                    log_dict.update({"train/lr": self.scheduler.get_last_lr()[0]})
                if valid_flag:
                    log_dict.update({"eval/valid_loss": valid_loss})
                if test_flag:
                    for idx, mtd in enumerate(kwargs['test_methods']):
                        log_dict.update({f"eval/train_{mtd['name']}": train_score[idx]})
                        if valid_flag:
                            log_dict.update({f"eval/valid_{mtd['name']}": valid_score[idx]})
                if best_score_flag:
                    log_dict.update({f"eval/best_valid_{kwargs['measure_best']}": best_score})

                wandb.log(log_dict)

            # update epoch
            epoch = epoch + 1

        total_time = time.time() - total_start_time

        # final message
        if self.device == 0:
            self.logger.info(f"---------- End of training. Total time: {round(total_time, 5)} seconds ----------")

        # reset config for next run
        self.save_dir = None
        self.logger = None
        self.last_best_epoch = -1


    def test_epoch(
        self, 
        loader: DataLoader, 
        methods: list[dict],
        resume_path: str | None = None, 
    ) -> tuple[list[float], str]:
        """test an epoch on given data by multiple given score methods

        will not update model parameters

        Parameters
        ----------
        loader: DataLoader
        methods: [
            {
                'name': str
                'function': function(out, label)
                'do_avg': bool
                    # scores are computed batch by batch, then summed up
                    # if True, final score will be averaged by size of given dataset
            }, 
            ......
        ]
        resume_path: str | None = None
            # if provided, test on provided model
            # otherwise, test on current model

        Returns
        -------
        : list[float]
            # list of test results
        : str
            # test message
        """

        # assert config correctness
        assert type(methods) is list, \
            "Trainer.test_epoch(): requires "\
            "`methods`: list[dict]"
        req_kws = ['name', 'function', 'do_avg']
        for mtd in methods:
            assert all(kw in mtd.keys() for kw in req_kws) \
                and type(mtd['name']) is str \
                and type(mtd['do_avg']) is bool, \
                "Trainer.test_epoch(): items in `methods` should be: "\
                "{'name': str, 'function': function(out, label), 'do_avg': bool}"
        if resume_path is not None:
            assert os.path.isfile(resume_path), \
                "Trainer.test_epoch(): `resume_path` not exists or not a file"

        # load checkpoint if provided
        if resume_path is not None:
            _ = self._resume_checkpoint(resume_path=resume_path)
        
        test_results = [0] * len(methods)

        self.model.eval()
        
        with torch.no_grad():
            for features, labels in tqdm(loader):
                # move to device
                features = features.to(self.device)
                labels = labels.to(self.device)

                # forward pass
                out = self.model(features)

                # enumerate all given methods to test score
                for idx, mtd in enumerate(methods):
                    score = mtd['function'](out, labels)
                    test_results[idx] += score

        # do average
        for idx, mtd in enumerate(methods):
            if mtd['do_avg'] is True:
                # average by size of dataset
                test_results[idx] /= len(loader.dataset)
            # generate message
            if idx == 0:
                test_msg = f"{mtd['name']}: {test_results[idx]}"
            else:
                test_msg = f"{test_msg} {mtd['name']}: {test_results[idx]}"

        return test_results, test_msg
    

    def _train_epoch(self, epoch: int) -> float:
        """train an epoch on `self.train_loader`

        will update model parameters

        Returns
        -------
        : float
            # loss of the last batch
        """
        # to make shuffling work properly across multiple epochs
        # otherwise, the same ordering will be used in each epoch
        self.train_loader.sampler.set_epoch(epoch)

        self.model.train()

        for features, labels in tqdm(self.train_loader):
            # to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # forward pass
            out = self.model(features)
            
            # compute loss
            loss = self.criterion(out, labels)
            
            # remove gradient from previous passes
            self.optimizer.zero_grad()
            
            # backprop
            loss.backward()
            
            # parameters update
            self.optimizer.step()
    
        return loss.item()
    

    def _valid_epoch(self) -> float:
        """validate an epoch on `self.valid_loader`

        will not update model parameters
        
        Returns
        -------
        : float
            # loss of the last batch
        """
        self.model.eval()

        with torch.no_grad():
            for features, labels in tqdm(self.valid_loader):
                # to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # forward pass
                out = self.model(features)

                # compute loss
                loss = self.criterion(out, labels)
                    
        return loss.item()
    

    def _set_logger(self, save_log: bool = False):
        """set logger for trainer

        stream handler set by default, file handler set if required

        """
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', 
                                      datefmt='%Y-%m-%d %H:%M:%S')
        if save_log is True:
            filename = os.path.join(self.save_dir, 'train.log')
            fileHandler = logging.FileHandler(filename)
            fileHandler.setFormatter(formatter)
            logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)

        self.logger = logger


    def _save_checkpoint(self, epoch: int, save_mode: str | None = None):
        """save checkpoint for best model, latest epoch, or specified epoch

        """
        model_class_name = type(self.model).__name__

        checkpoint = {
            'model_class_name': model_class_name,
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),  # model has been wrapped by DDP
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        if save_mode == 'best':    # only keep one best model
            del_filename = os.path.join(self.save_dir, f'best_model_epoch{self.last_best_epoch}.pth')
            if os.path.exists(del_filename):
                os.remove(del_filename)
            filename = os.path.join(self.save_dir, f'best_model_epoch{epoch}.pth')
            torch.save(checkpoint, filename)
            self.logger.info(f"Saving best model: {filename} ...")
            self.last_best_epoch = epoch

        if save_mode == 'latest':    # only keep one latest checkpoint, and no log for it
            del_filename = os.path.join(self.save_dir, f'latest_checkpoint_epoch{epoch-1}.pth')
            if os.path.exists(del_filename):
                os.remove(del_filename)
            filename = os.path.join(self.save_dir, f'latest_checkpoint_epoch{epoch}.pth')
            torch.save(checkpoint, filename)

        if save_mode is None:
            filename = os.path.join(self.save_dir, f'checkpoint_epoch{epoch}.pth')
            torch.save(checkpoint, filename)
            self.logger.info(f"Saving checkpoint: {filename} ...")


    def _resume_checkpoint(self, resume_path: str) -> int:
        """resume checkpoint

        Returns
        -------
        : int
            # start epoch
        """
        if self.logger is not None:
            self.logger.info(f"Loading checkpoint: {resume_path} ...")
        else:
            print(f"Loading checkpoint: {resume_path} ...")

        checkpoint = torch.load(resume_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        if self.logger is not None:
            self.logger.info(f"Checkpoint loaded. Resume training from epoch {start_epoch}")
        else:
            print("Checkpoint loaded.")

        return start_epoch