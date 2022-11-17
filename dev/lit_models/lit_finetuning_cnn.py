"""LitModule for finetuning image models."""
import argparse
import yaml

import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from pytorch_metric_learning import losses

import dev.metadata.inat as metadata

## load config
config_file = "training_config.yml"
with open(config_file, "rb") as file:
        config = yaml.load(file, Loader=yaml.Loader)

### lightning module for finetuning CNN
class LitFinetuningCNN(pl.LightningModule):

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        
        self.save_hyperparameters(ignore=['model'])
        
        # set up model and optimizer hps
        self.model = model
        self.args = vars(args) if args is not None else {}
        self.data_config = self.model.data_config
        self.input_dims = self.data_config["input_dims"]

        optimizer = self.args.get("optimizer", config['OPTIMIZER'])
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", float(config['LR']))
        self.wd = self.args.get("wd", float(config['WEIGHT_DECAY']))

        self.loss = self.args.get("loss", config['LOSS'])
        if self.loss == "arcface":
            self.loss_fn = losses.ArcFaceLoss(num_classes=metadata.NUM_PLANT_CLASSES,
                                              embedding_size=metadata.NUM_PLANT_CLASSES)
        else:
            self.loss_fn = getattr(torch.nn.functional, self.loss)
            
        # use accuracy as the gold star metric
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr, weight_decay=self.wd)
        
        # vanilla optimizer config
        if not config['REDUCE_LR_ON_PLATEAU'] and not config['ONE_CYCLE_LR']:
            return optimizer
        
        # optimizer config for reduce lr on plateau
        if config['REDUCE_LR_ON_PLATEAU']:
            assert self.one_cycle_max_lr is None, "Error: 1cycle scheduler incompatible with ReduceLROnPlateau"
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                            factor=config['ROP_COEFF'],
                                                            patience=0, verbose=True,
                                                            threshold=config['ROP_THRESHOLD'],
                                                            threshold_mode=config['ROP_THRESHOLD_MODE'],
                                                            min_lr=2e-5,
            )
            
        # optimizer config for 1cycle lr schedule
        if config['ONE_CYCLE_LR']:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=config['ONE_CYCLE_MAX_LR'],
                total_steps=config['ONE_CYCLE_TOTAL_STEPS'],
                pct_start=0.1,
            )
            
        return {
            "optimizer" : optimizer,
                "lr_scheduler": {
                    "scheduler" : scheduler,
                    "monitor": "validation/acc"
            }
        }

    ## basic lightning overrides
    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        logits = self.model(x)
        return torch.argmax(logits, dim=1)

    def training_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        self.train_acc(logits, y)
        preds = self.get_preds(logits)

        self.log("train/loss", loss)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)

        outputs = {"loss": loss}
        self.add_on_first_batch({"preds": preds, "labels": y}, outputs, batch_idx)

        return outputs

    def validation_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        self.val_acc(logits, y)

        # TODO: on mps, I get an error when using wandb+lighting b/c
        # metadata={'score': tensor(0.0049, device='mps:0')
        # gets passed to _normalize_metadata in wandb_artifact(s).py
        self.log("validation/loss", loss, prog_bar=True, sync_dist=True)
        self.log("validation/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        outputs = {"loss": loss}

        self.add_on_first_batch({"logits": logits.detach()}, outputs, batch_idx) # added

        return outputs
    
    def test_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        self.test_acc(logits, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

    ## other lightning overrides
    def _run_on_batch(self, batch, with_preds=False):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        return x, y, logits, loss

    def add_on_first_batch(self, metrics, outputs, batch_idx):
        if batch_idx == 0:
            outputs.update(metrics)

    def add_on_logged_batches(self, metrics, outputs):
        if self.is_logged_batch:
            outputs.update(metrics)

    def is_logged_batch(self):
        if self.trainer is None:
            return False
        else:
            return self.trainer._logger_connector.should_update_logs

    def get_preds(self, logitlikes: torch.Tensor, replace_after_end: bool = True) -> torch.Tensor:
        """Converts logit-like Tensors into prediction indices, optionally overwritten after end token index.

        Parameters
        ----------
        logitlikes
            (B, C) Tensor with classes as second dimension. The largest value is the one
            whose index we will return. Logits, logprobs, and probs are all acceptable.
        
        Returns
        -------
        torch.Tensor
            (B) Tensor of integers in [0, C-1] representing predictions.
        """
        raw = torch.argmax(logitlikes, dim=1)  # (B, C) -> (B,)
        return raw  # (B,)
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=config['OPTIMIZER'], help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=config['LR'])
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=config['ONE_CYCLE_TOTAL_STEPS'])
        parser.add_argument("--loss", type=str, default=config['LOSS'], help="loss function from torch.nn.functional")
        return parser