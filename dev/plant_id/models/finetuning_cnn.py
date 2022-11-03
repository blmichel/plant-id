"""Basic convolutional model building blocks."""
import argparse
from typing import Any, Dict

import math
import torch
from torch import nn
import torch.nn.functional as F
import timm
from timm.models.layers.norm_act import BatchNormAct2d
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d

import plant_id.metadata.inat as metadata

import yaml
config_file = "../training/training_config.yml"
with open(config_file, "rb") as file:
        config = yaml.load(file)

## TODO: integrate with old
## TODO: hardcoded for now
FC_DIM = metadata.NUM_PLANT_CLASSES
### torch module for CNN to finetune
# TODO: adapt for geodata and vision transformers
class FinetuningCNN(nn.Module):
    """Load a pretrained CNN and stick on a classifier for finetuning."""

    def __init__(self, data_config: Dict[str, Any], model_config: Dict[str, Any],
                 args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config
        self.model_config = model_config

        input_channels, input_height, input_width = self.data_config["input_dims"]
        assert (
            input_height == input_width
        ), f"input height and width should be equal, but was {input_height}, {input_width}"
        self.input_height, self.input_width = input_height, input_width
        
        # when repackaging, want eg num_classes = metadata.NUM_PLANT_CLASSES
        num_classes = metadata.NUM_PLANT_CLASSES
        if args is not None:
            # TODO: where is it getting FC_DIM from??
            fc_dim = self.args.get("fc_dim", FC_DIM)
            fc_dropout = self.args.get("fc_dropout", config['FC_DROPOUT'])
            pretrained_stem = self.args.get("pretrained_stem", pretrained_stem)
        else:
            pretrained_stem = model_config["pretrained_stem"]
            fc_dim = model_config["fc_dim"]
            fc_dropout = model_config["fc_dropout"]

        self.mode = model_config["mode"]
        
        ## load pretrained model
        m = timm.create_model(pretrained_stem, pretrained=True, num_classes=fc_dim)

        ## get pretrained stem and define architecture-specific head layers
        if self.mode == 'resnet':
            self.stem = nn.Sequential(*[i for i in m.children()][:-2])
            stem_out_feats = self.stem[-1][-1].bn3.num_features
            # the resnet stem contains a final batchnorm so only need to normalize
            # after the depthwise seperable conv
            self.bn = BatchNormAct2d(
                fc_dim, 
                eps=0.001, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True,
                drop_layer = None,
                act_layer = torch.nn.SiLU
            )
        elif self.mode == 'lambda':
            self.stem = nn.Sequential(*[i for i in m.children()][:-2])
            stem_out_feats = 2048
            self.bn = BatchNormAct2d(
                fc_dim, 
                eps=0.001, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True,
                drop_layer = None,
                act_layer = torch.nn.SiLU
            )
        elif self.mode == 'efficientnet':
            self.stem = m.conv_stem
            self.bn1 = m.bn1
            self.blocks = m.blocks
            # the below is ugly, would be nice to define it elsewhere
            if 'v2_m' in pretrained_stem:
                stem_out_feats = 512
            elif 'v2_s' in pretrained_stem:
                stem_out_feats = 256
            else:
                # what model is this?
                stem_out_feats = 640        
            self.bn2 = BatchNormAct2d(
                stem_out_feats, 
                eps=0.001, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True,
                drop_layer = None,
                act_layer = torch.nn.SiLU
            )                
            self.bn3 = BatchNormAct2d(
                fc_dim, 
                eps=0.001, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True,
                drop_layer = None,
                act_layer = torch.nn.SiLU
            )
            
        elif self.mode == 'convnext':
            self.stem = m.stem
            self.stages = m.stages # after, shape = [bs, 1024, 7, 7]
            if 'large' in pretrained_stem:
                stem_out_feats = 1536
            if 'base' in pretrained_stem:
                stem_out_feats = 1024
            elif 'small' in pretrained_stem:
                stem_out_feats = 768
            elif 'tiny' in pretrained_stem:
                stem_out_feats = 768
                
        elif self.mode == 'resnext':
            self.stem = nn.Sequential(*[i for i in m.children()][:-2])
            stem_out_feats = 2048
            self.bn = BatchNormAct2d(
                fc_dim, 
                eps=0.001, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True,
                drop_layer = None,
                act_layer = torch.nn.SiLU
            )
            
        else:
            ValueError("Currently supported model types: (lambda)-resne(x)t, \
                efficientnet(v2_m/s), convnext_(l/b/s/t)")
        
        ## head layers are shared across base architectures    
        if fc_dropout != 0:
            self.dropout = nn.Dropout(fc_dropout)
        else:
            self.dropout = nn.Identity()
        self.pool = SelectAdaptivePool2d(pool_type = 'avg', flatten = True)

        if config['HEAD_DWSCONV']:
            self.dwsconv = torch.nn.Conv2d(stem_out_feats, fc_dim, kernel_size = 1, stride = 1)
            # TODO: this is just the pointwise part of the dwsconv. add dwise
            self.classifier = torch.nn.Linear(fc_dim, num_classes)
            ln_feats = fc_dim
            self._init_weights(self.dwsconv)
        else:
            self.classifier = torch.nn.Linear(stem_out_feats, num_classes)
            ln_feats = stem_out_feats
            
        if self.mode == 'convnext':
            self.ln1 = nn.LayerNorm(ln_feats, eps=1e-06, elementwise_affine=True)
        
        self._init_weights(self.classifier)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the CNN to x.

        Parameters
        ----------
        x
            (B, Ch, H, W) tensor, where H and W must equal input height and width from data_config.

        Returns
        -------
        torch.Tensor
            (B, Cl) tensor
        """
        _B, _Ch, H, W = x.shape
        assert H == self.input_height and W == self.input_width, f"bad inputs to CNN with shape {x.shape}"
        # TODO: process by iNatStem instead of doing transforms in __getitem__?
        x = self.stem(x)
        
        if self.mode == 'resnet' or self.mode == 'lambda':
            # experiment with dropout layer
            x = self.dropout(x)
            x = self.dwsconv(x)
            # experiment with batchnorm layer here
            x = self.bn(x)
            x = self.pool(x)
            x = self.classifier(x)
        if self.mode == 'resnext':
            # experiment with dropout layer
            x = self.dropout(x)
            x = self.dwsconv(x)
            # experiment with batchnorm layer here
            x = self.bn(x)
            x = self.pool(x)
            x = self.classifier(x)
        elif self.mode =='efficientnet':
            x = self.bn1(x)
            x = self.blocks(x)
            x = self.bn2(x)
            # put a dropout layer here?
            # experimenting...
            x = self.dropout(x)
            x = self.dwsconv(x)
            x = self.bn3(x)
            x = self.pool(x)
            x = self.classifier(x)
        elif self.mode == 'convnext':
            x = self.stages(x)
            # experimenting with removing dwsconv
            if config['HEAD_DWSCONV']:
                x = self.dwsconv(x)
            # could put dropout here
            x = x.permute(0, 2, 3, 1) # channels last for layernorm
            x = self.ln1(x)
            x = x.permute(0, 3, 1, 2) # channels first for pool
            x = self.pool(x)
            x = self.classifier(x)
        return x

    def _init_weights(self, m):
        """
        Initialize weights in a better way than default.
        See https://github.com/pytorch/pytorch/issues/18182
        """
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(m.bias, -bound, bound)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        parser.add_argument("--fc_dropout", type=float, default=config['FC_DROPOUT'])
        parser.add_argument("--pretrained_model", type=str, default=pretrained_stem)
        return parser



### old -- think you can delete everything below
#FC_DIM = 128
#FC_DROPOUT = 0.25
#MODEL_NAME = 'resnet50'

class Pretrained_CNN(nn.Module):
    """
    Loads a pretrained CNN from timm
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        # TODO: is this getting to the GPU via args passed to the Lightning Trainer?
        try:
            self.pretrained = timm.create_model(model_name, pretrained=True, num_classes = FC_DIM)#.to(device)
        except Exception as e:
            print(e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the ConvBlock to x.

        Parameters
        ----------
        x
            (B, C, H, W) tensor

        Returns
        -------
        torch.Tensor
            (B, FC_DIM) tensor
        """
        return self.pretrained(x)


class Finetuning_CNN(nn.Module):
    """Load a pretrained CNN and stick on a classifier for finetuning."""

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config

        input_channels, input_height, input_width = self.data_config["input_dims"]
        assert (
            input_height == input_width
        ), f"input height and width should be equal, but was {input_height}, {input_width}"
        self.input_height, self.input_width = input_height, input_width

        num_classes = metadata.NUM_PLANT_CLASSES

        fc_dim = self.args.get("fc_dim", FC_DIM)
        fc_dropout = self.args.get("fc_dropout", FC_DROPOUT)
        model_name = self.args.get("model_name", MODEL_NAME)
        
        self.pretrained_model = Pretrained_CNN(model_name)
        self.dropout = nn.Dropout(fc_dropout)
        self.classifier = nn.Linear(fc_dim, num_classes)
        self._init_weights(self.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the CNN to x.

        Parameters
        ----------
        x
            (B, Ch, H, W) tensor, where H and W must equal input height and width from data_config.

        Returns
        -------
        torch.Tensor
            (B, Cl) tensor
        """
        _B, _Ch, H, W = x.shape
        assert H == self.input_height and W == self.input_width, f"bad inputs to CNN with shape {x.shape}"
        x = self.pretrained_model(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _init_weights(self, m):
        """
        Initialize weights in a better way than default.
        See https://github.com/pytorch/pytorch/issues/18182
        """
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(m.bias, -bound, bound)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        parser.add_argument("--fc_dropout", type=float, default=FC_DROPOUT)
        parser.add_argument("--model_name", type=str, default=MODEL_NAME)
        return parser

