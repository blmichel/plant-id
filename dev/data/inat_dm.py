"""DataModule class for Plant ID."""
import argparse
import yaml
import os
import multiprocessing

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dev import util
from dev.data.inat_ds import iNatDataset
import dev.metadata.inat as metadata

## from FSDL: sensible multiprocessing defaults, at most one worker per CPU
if torch.cuda.is_available():
    NUM_AVAIL_CPUS = len(os.sched_getaffinity(0))
    NUM_AVAIL_GPUS = torch.cuda.device_count()
# on mps, can't use sched_getaffinity
elif torch.backends.mps.is_available():
    # hardcoded for now
    NUM_AVAIL_CPUS = 8
    NUM_AVAIL_GPUS = 1
# if no GPU
else:
    NUM_AVAIL_CPUS = multiprocessing.cpu_count()
    NUM_AVAIL_GPUS = 1
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
# in distributed data parallel mode, we launch a training on each GPU, so must divide out to keep total at one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS

## load config
config_file = "training_config.yml"
with open(config_file, "rb") as file:
        config = yaml.load(file, Loader=yaml.Loader)

## get appropriate directories from metadata 
if config['TRAIN_SET'] == 'mini':
    TRAIN_DIR = metadata.MINI_DATA_DIRNAME
elif config['TRAIN_SET'] == 'full':
    TRAIN_DIR = metadata.TRAIN_DATA_DIRNAME
VAL_DIR = metadata.VAL_DATA_DIRNAME
#TEST_DIR = metadata.TEST_DATA_DIRNAME   


### lightning datamodule wrapper for iNatDataset
class iNatDataModule(pl.LightningDataModule):
    
    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", config['BATCH_SIZE'])
        self.num_workers = self.args.get("num_workers", DEFAULT_NUM_WORKERS)
        self.train_dir = self.args.get("train_dir", TRAIN_DIR)
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))
        
    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {"input_dims": self.input_dims}

    # Resize and ToTensor-only transform currently hardcoded (alt: advanced_transform)
    # TODO: should we normalize in util.util.transform?
    def setup(self, stage=None):
        self.data_train = iNatDataset(TRAIN_DIR, transform = util.util.transform())
        self.data_val = iNatDataset(VAL_DIR, transform = util.util.transform())
        # self.data_test 
        
    def prepare_data(self):  # prepares state that needs to be set once per node
        pass  # but we don't have any "node-level" computations

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )    
    
    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )
        
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size",
            type=int,
            default=config['BATCH_SIZE'],
            help=f"Number of examples to operate on per forward step. Default is {config['BATCH_SIZE']}.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=DEFAULT_NUM_WORKERS,
            help=f"Number of additional processes to load data. Default is {DEFAULT_NUM_WORKERS}.",
        )
        return parser   