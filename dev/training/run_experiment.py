"""Experiment-running framework."""
import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
import torch

from plant_id import callbacks as cb

from plant_id import lit_models
from training.util import DATA_CLASS_MODULE, import_class, MODEL_CLASS_MODULE, setup_data_and_model_from_args


# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


# this is all old, until the constants defined before the second main
def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])
    parser.set_defaults(max_epochs=1)

    # Basic arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=True,
        help="If passed, logs experiment results to Weights & Biases. Otherwise logs only to local Tensorboard.",
    )
    parser.add_argument(
        "--data_class",
        type=str,
        default="INAT_MINI",
        help=f"String identifier for the data class, relative to {DATA_CLASS_MODULE}.",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="PRETRAINED_CNN",
        help=f"String identifier for the model class, relative to {MODEL_CLASS_MODULE}.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        help=f"String identifier for the pretrained model to load from timm.",
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default=None, help="If passed, loads a model from the provided path."
    )
    parser.add_argument(
        "--stop_early",
        type=int,
        default=0,
        help="If non-zero, applies early stopping, with the provided value as the 'patience' argument."
        + " Default is 0.",
    )

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = import_class(f"{DATA_CLASS_MODULE}.{temp_args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


@rank_zero_only
def _ensure_logging_dir(experiment_dir):
    """Create the logging directory via the rank-zero process, if necessary."""
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)


def main_from_fsdl():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=MLP --data_class=MNIST
    ```

    For basic help documentation, run the command
    ```
    python training/run_experiment.py --help
    ```

    The available command line args differ depending on some of the arguments, including --model_class and --data_class.

    To see which command line args are available and read their documentation, provide values for those arguments
    before invoking --help, like so:
    ```
    python training/run_experiment.py --model_class=MLP --data_class=MNIST --help
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data, model = setup_data_and_model_from_args(args)

    lit_model_class = lit_models.BaseLitModel

    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)

    log_dir = Path("training") / "logs"
    _ensure_logging_dir(log_dir)
    logger = pl.loggers.TensorBoardLogger(log_dir)
    experiment_dir = logger.log_dir

    goldstar_metric = "validation/acc"
    filename_format = "epoch={epoch:04d}-validation.acc={validation/acc:.3f}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        filename=filename_format,
        monitor=goldstar_metric,
        mode="min",
        auto_insert_metric_name=False,
        dirpath=experiment_dir,
        every_n_epochs=args.check_val_every_n_epoch,
    )

    summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    callbacks = [summary_callback, checkpoint_callback]
    if args.wandb:
        logger = pl.loggers.WandbLogger(log_model="all", save_dir=str(log_dir), job_type="train")
        logger.watch(model, log_freq=max(100, args.log_every_n_steps))
        logger.log_hyperparams(vars(args))
        experiment_dir = logger.experiment.dir
    callbacks += [cb.ModelSizeLogger(), cb.LearningRateMonitor()]
    if args.stop_early:
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="validation/loss", mode="min", patience=args.stop_early
        )
        callbacks.append(early_stopping_callback)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)

    trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)

    trainer.test(lit_model, datamodule=data)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        rank_zero_info(f"Best model saved at: {best_model_path}")
        if args.wandb:
            rank_zero_info("Best model also uploaded to W&B ")

# previously
#if __name__ == "__main__":
#    main()
### main



### here's the start of your stuff
# TODO: load stuff from config yaml

### hardcoded stuff - will want to add some to CLI later...

import plant_id.metadata.inat as metadata
import yaml
config_file = "training_config.yml"
with open(config_file, "rb") as file:
        config = yaml.load(file)

#if config['TRAIN_SET'] == 'mini':
#    TRAIN_DIR = 'data_2021_mini/2021_train_mini'
#elif config['TRAIN_SET'] == 'full':
#    TRAIN_DIR = 'data_2021_full/train'


# generate run name
WANDB_RUN_NAME = f"{config['PRETRAINED_STEM']} \
    wd={config['WEIGHT_DECAY']}"
if config['REDUCE_LR_ON_PLATEAU']:
    WANDB_RUN_NAME += f" ROP={config['ROP_COEFF']}"
if config['LAYERWISE_LR_DECAY']:
    WANDB_RUN_NAME += f" LLRD={config['LLRD_COEFF']}"
if not config['HEAD_DWSCONV']:
    WANDB_RUN_NAME += ' head_dws=False'
if config['TRAIN_SET'] == 'full':
    WANDB_RUN_NAME += ' ds=full'

# could do control flow here to add options only when they're non-default


### imports and setup
# TODO: hardcoded for now
FC_DIM = metadata.NUM_PLANT_CLASSES
# TODO: remove unnecessary imports
import sys
sys.path.append('../utils')
sys.path.insert(1, '/home/team_050/plantid-env/lib/python3.8/site-packages')
if '/usr/lib/python3/dist-packages' in sys.path:
    sys.path.remove('/usr/lib/python3/dist-packages')
    sys.path.remove('/home/team_050/.local/lib/python3.8/site-packages')

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import utils
from torchvision import transforms as T
from torchvision.transforms import ToTensor, Resize
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy

from pathlib import Path
import glob
import pdb
import argparse
from tqdm import tqdm
from typing import Callable
from PIL import Image
import os
import math
import numpy as np
from typing import Any, Dict

import timm
import wandb
import pytorch_lightning as pl

from timm.models.layers.norm_act import BatchNormAct2d
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d

import plant_id.models as models
import plant_id.data as data
import plant_id.lit_models as lit_models


### from FSDL: sensible multiprocessing defaults: at most one worker per CPU
NUM_AVAIL_CPUS = len(os.sched_getaffinity(0))
NUM_AVAIL_GPUS = torch.cuda.device_count()
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
# but in distributed data parallel mode, we launch a training on each GPU, so must divide out to keep total at one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS





def main(pretrained_stem, use_wandb=True):
    
    print("Pretrained base model: ", pretrained_stem)
    
    ## different models have different names for stem and blocks
    if 'lambda' in pretrained_stem:
        mode = 'lambda'
    elif 'resnet' in pretrained_stem:
        mode = 'resnet'
    elif 'efficientnet' in pretrained_stem:
        mode = 'efficientnet'
    elif 'convnext' in pretrained_stem:
        mode = 'convnext'
    elif 'resnext' in pretrained_stem:
        mode = 'resnext'
    
    ## set log dir and format filename
    log_dir = Path("training") / "logs"
    logger = pl.loggers.TensorBoardLogger(log_dir)
    experiment_dir = logger.log_dir

    goldstar_metric = "validation/acc"
    filename_format = "epoch={epoch:04d}-validation.loss={validation/loss:.3f}"
    if goldstar_metric == "validation/acc":
        filename_format += "-validation.acc={validation/acc:.3f}"
    
    ## callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=2,
        filename=filename_format,
        monitor=goldstar_metric,
        mode="max",
        auto_insert_metric_name=False,
        dirpath=experiment_dir,
    )
    lr_callback = pl.callbacks.LearningRateMonitor()
    callbacks = [checkpoint_callback, lr_callback, ]
    if config['EARLY_STOPPING']:
        callbacks.append(pl.callbacks.EarlyStopping(
        monitor="validation/acc",
        mode="max",
        patience=1,
        ))
    if config['STOCHASTIC_WEIGHT_AVERAGING']:
        callbacks.append(pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-4,
                                                                swa_epoch_start=0.5,
                                                                annealing_epochs=5,
        ))
    
    ## set up lit model and datamodule
    DATA_CONFIG = {"input_dims" : (3, metadata.RESOLUTION, metadata.RESOLUTION)}
    MODEL_CONFIG = {"pretrained_stem" : pretrained_stem, "fc_dim": FC_DIM,
                    "fc_dropout" : config['FC_DROPOUT'], "mode": mode}    

    model = models.FinetuningCNN(data_config=DATA_CONFIG, model_config=MODEL_CONFIG)
    lit_model = lit_models.LitFinetuningCNN(model)
    datamodule = data.iNatDataModule()
        
    if config['LOADED_MODEL']:
        loaded_lit_model = torch.load(config['LOADED_MODEL'])
        #TODO: replace with load_state_dict, this currently doesn't work
        lit_model.state_dict = loaded_lit_model.state_dict # need this step for wandb to log
        del(loaded_lit_model)
        
    ## optionally, watch model with wandb    
    if use_wandb: #args.wandb
        wandb.init(
            project=config['WANDB_PROJECT_NAME'],
            notes="testing wandb integration",
            name=WANDB_RUN_NAME,
            tags=["test"],
            config=config,
        )
        logger = pl.loggers.WandbLogger(log_model="all", save_dir=str(log_dir), job_type="train")
        logger.watch(lit_model)
        experiment_dir = logger.experiment.dir

    ## training code
    trainer = pl.Trainer(max_epochs=config['NUM_EPOCHS'],
                         devices=config['DEVICES'], accelerator='gpu',
                         callbacks=callbacks, logger=logger,
                         auto_scale_batch_size=config['AUTO_SCALE_BATCH_SIZE'],
                         auto_lr_find=config['AUTO_LR_FIND'],
                         precision=config['PRECISION'],
                         limit_train_batches=config['LIMIT_TRAIN_BATCHES'],
                         )
    trainer.tune(lit_model, datamodule=datamodule)
    trainer.fit(lit_model, datamodule=datamodule)
    
    ## save the model and state dict to disk and add as artifact to wandb
    # TODO: only save if val/acc above some threshold
    if use_wandb:
        run_id = wandb.run.id
    else:
        run_id = 'not_saved_to_wandb' # add timestamp here?
    wandb.unwatch(lit_model) # need to remove wandb hooks in order to torch.save model
    torch.save(lit_model, f"model_{run_id}.pkl")    
    model_pkl_name = f"trained_model_{run_id}" 
    model_artifact = wandb.Artifact(model_pkl_name, "model")    
    model_artifact.add_file(f"model_{run_id}.pkl")    
    wandb.log_artifact(model_artifact)
    
    if use_wandb:
        wandb.finish()

### current usage: python run_experiment.py True
if __name__ == '__main__':
    import sys
    
    ## load args
    use_wandb = sys.argv[1].lower() == 'true'

    ## process args
    if config['AUTO_LR_FIND']:
        lr = 'autotune'
    else:
        lr = config['LR']
        
    if config['AUTO_SCALE_BATCH_SIZE']==None:
        auto_scale_batch_size = 'False'
    else:
        auto_scale_batch_size = config['AUTO_SCALE_BATCH_SIZE']
    
    if config['LOADED_MODEL==None']:
        loaded_model = 'None'
    else:
        loaded_model = config['LOADED_MODEL']
        
    ## wandb metadata    
    config = dict (
        dataset_id = "iNat-2021",
        infra = "Lambda VM",
        pretrained_stem = config['PRETRAINED_STEM'],
        resolution = config['RESOLUTION'],
        learning_rate = lr,
        batch_size = config['BATCH_SIZE'], # need better logging: if autoscaled, this isn't the actual bs
        auto_scale_batch_size = auto_scale_batch_size,
        weight_decay = config['WEIGHT_DECAY'],
        fc_dropout = config['FC_DROPOUT'],
        fc_dim = FC_DIM,
        precision = config['PRECISION'],
        limit_train_batches = config['LIMIT_TRAIN_BATCHES'],
        dataset_type = config['TRAIN_SET'],
        loaded_model = loaded_model,
        reduce_lr_on_plateau = config['REDUCE_LR_ON_PLATEAU'],
        use_swa = config['STOCHASTIC_WEIGHT_AVERAGING'],
        early_stopping = config['EARLY_STOPPING'],
        loss = config['LOSS'],
        layerwise_lr_decay = config['LAYERWISE_LR_DECAY'],
        layerwise_lr_decay_coeff = config['LLRD_COEFF'],
        head_dwsconv = config['HEAD_DWSCONV'],
        rop_coeff = config['ROP_COEFF'],
        rop_threshold = config['ROP_THRESHOLD'],
        rop_threshold_mode = config['ROP_THRESHOLD_MODE'],
    )    
       
    ## run
    main(config['PRETRAINED_STEM'], use_wandb)