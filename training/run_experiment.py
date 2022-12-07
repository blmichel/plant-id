"""Experiment-running framework."""
from pathlib import Path
import yaml
import sys
import os

import numpy as np
import pytorch_lightning as pl
import torch
import wandb

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import dev.lit_models as lit_models
import dev.models as models
import dev.data as data

## fix random seed
np.random.seed(42)
torch.manual_seed(42)

## load training config
config_file = "training_config.yml"
with open(config_file, "rb") as file:
        config = yaml.load(file, Loader=yaml.Loader)

## generate run name
WANDB_RUN_NAME = f"{config['PRETRAINED_STEM']} wd={config['WEIGHT_DECAY']}"
if config['REDUCE_LR_ON_PLATEAU']:
    WANDB_RUN_NAME += f" ROP={config['ROP_COEFF']}"
if config['LAYERWISE_LR_DECAY']:
    WANDB_RUN_NAME += f" LLRD={config['LLRD_COEFF']}"
if not config['HEAD_DWSCONV']:
    WANDB_RUN_NAME += ' head_dws=False'
if config['TRAIN_SET'] == 'full':
    WANDB_RUN_NAME += ' ds=full'
# could do control flow here to add options only when they're non-default

### main
def main(training_config):
    
    pretrained_stem = training_config['pretrained_stem']
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
    # TODO: create if not there already
    #if not os.path.isdir("../data/transport")
    log_dir = Path.cwd() / "logs"
    logger = pl.loggers.TensorBoardLogger(log_dir)
    experiment_dir = logger.log_dir

    goldstar_metric = "validation/acc"
    filename_format = "epoch={epoch:04d}-validation.loss={validation/loss:.3f}"
    if goldstar_metric == "validation/acc":
        filename_format += "-validation.acc={validation/acc:.3f}"
    
    ## set up callbacks
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
        monitor=goldstar_metric,
        mode="max",
        patience=1,
        ))
    if config['STOCHASTIC_WEIGHT_AVERAGING']:
        callbacks.append(pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-4,
                                                                swa_epoch_start=0.5,
                                                                annealing_epochs=5,
        ))
    
    ## configure module options
    input_channels = 3
    DATA_CONFIG = {"input_dims" : (input_channels, training_config['resolution'], training_config['resolution'])}
    MODEL_CONFIG = {"pretrained_stem" : pretrained_stem, "fc_dim": training_config['fc_dim'],
                    "fc_dropout" : training_config['fc_dropout'], "mode": mode}    

    ## set up lit model and datamodule
    model = models.FinetuningCNN(data_config=DATA_CONFIG, model_config=MODEL_CONFIG)
    lit_model = lit_models.LitFinetuningCNN(model)
    datamodule = data.iNatDataModule()
    
    ## optionally, load model from checkpoint
    if config['LOADED_MODEL'] != 'None':
        loaded_lit_model = torch.load(config['LOADED_MODEL'])
        lit_model.load_state_dict(loaded_lit_model.state_dict) # need this step for wandb to log
        del(loaded_lit_model)
        
    ## optionally, watch model with wandb    
    if config['USE_WANDB']:
        wandb.init(
            project=config['WANDB_PROJECT_NAME'],
            notes="testing wandb integration",
            name=WANDB_RUN_NAME,
            tags=["test"],
            config=training_config,
        )
        logger = pl.loggers.WandbLogger(log_model="all", save_dir=str(log_dir), job_type="train")
        logger.watch(lit_model)
        experiment_dir = logger.experiment.dir

    ## launch training
    trainer = pl.Trainer(max_epochs=config['NUM_EPOCHS'],
                         devices=config['DEVICES'],
                         accelerator=config['ACCELERATOR'],
                         callbacks=callbacks,
                         logger=logger,
                         auto_scale_batch_size=auto_scale_batch_size,
                         auto_lr_find=config['AUTO_LR_FIND'],
                         precision=training_config['precision'],
                         limit_train_batches=config['LIMIT_TRAIN_BATCHES'],
                         )
    trainer.tune(lit_model, datamodule=datamodule)
    # need to update wandb config if tuning bs or lr
    if config['USE_WANDB']:
        wandb.config.update({'batch_size': datamodule.batch_size})
        wandb.config.update({'lr': lit_model.lr})
    trainer.fit(lit_model, datamodule=datamodule)
    
    ## save the model to disk and add as artifact to wandb
    # TODO: only save if val/acc above some threshold 
    if config['USE_WANDB']:
        run_id = wandb.run.id
        wandb.unwatch(lit_model) # need to remove wandb hooks in order to torch.save model
    else:
        run_id = 'not_saved_to_wandb' # add timestamp here?

    torch.save(lit_model, f"model_{run_id}.pkl")    
    
    if config['USE_WANDB']:
        model_pkl_name = f"trained_model_{run_id}" 
        model_artifact = wandb.Artifact(model_pkl_name, "model")    
        model_artifact.add_file(f"model_{run_id}.pkl")
        
        config_artifact = wandb.Artifact(config_file, "config_yaml")
        config_artifact.add_file(config_file)  
          
        wandb.log_artifact(model_artifact)
        wandb.log_artifact(config_artifact)
        wandb.finish()

### usage: python3 run_experiment.py
if __name__ == '__main__':

    ## process config options
    if config['AUTO_LR_FIND']==True:
        lr = 'autotune'
        auto_lr_find = 'False'
    else:
        lr = config['LR']
        auto_lr_find = 'True'

    if config['AUTO_SCALE_BATCH_SIZE']=='None':
        auto_scale_batch_size = None
    else:
        auto_scale_batch_size = config['AUTO_SCALE_BATCH_SIZE']
    
    if config['LOADED_MODEL']==None:
        loaded_model = 'None'
    else:
        loaded_model = config['LOADED_MODEL']
    
    # TODO: use config yaml directly in wandb.init
    # to avoid ugliness like passing preprocessed auto_lr_find
    # but raw config['AUTO_SCALE_BATCH_SIZE']
    # (since the yaml loader can process bools but not nonetype)
        
    ## training metadata    
    training_config = dict(
        dataset_id = "iNat-2021",
        infra = "BLM MBA",
        pretrained_stem = config['PRETRAINED_STEM'],
        resolution = config['RESOLUTION'],
        lr = lr,
        auto_lr_find = auto_lr_find,
        batch_size = config['BATCH_SIZE'],
        auto_scale_batch_size = config['AUTO_SCALE_BATCH_SIZE'],
        weight_decay = config['WEIGHT_DECAY'],
        fc_dropout = config['FC_DROPOUT'],
        fc_dim = config['FC_DIM'],
        # AMP only available on CUDA
        precision = config['PRECISION'] if torch.cuda.is_available() else 32,
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
        one_cycle_lr = config['ONE_CYCLE_LR'],
    )    
       
    ## run
    main(training_config)