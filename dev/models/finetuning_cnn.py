"""Basic convolutional model building blocks."""
import argparse
from typing import Any, Dict
import yaml
import os

import math
import torch
from torch import nn
import torch.nn.functional as F
import timm
from timm.models.layers.norm_act import BatchNormAct2d
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d

import dev.metadata.inat as metadata

## load config
config_file = "training_config.yml"
with open(config_file, "rb") as file:
        config = yaml.load(file, Loader=yaml.Loader)


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
        
        num_classes = metadata.NUM_PLANT_CLASSES
        if args is not None:
            fc_dim = self.args.get("fc_dim", config['FC_DIM'])
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
            elif 'b7' in pretrained_stem:
                # what model is this? B7?
                stem_out_feats = 640   
            elif 'b0' in pretrained_stem:
                stem_out_feats = 320     
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

        # throw an error if pretrained_stem not yet supported   
        else:
            ValueError("Currently supported model types: (lambda)-resne(x)t, \
                efficientnet(v2_m/s), convnext_(l/b/s/t)")
        
        ## most head layers are shared across base architectures    
        if fc_dropout != 0:
            self.dropout = nn.Dropout(fc_dropout)
        else:
            self.dropout = nn.Identity()
        self.pool = SelectAdaptivePool2d(pool_type = 'avg', flatten = True)

        ## optionally, do a dwsconv on the feature maps output by the stem
        # I do the pointwise conv first, not sure how much order matters- 
        # this is slightly heavier weight since fc_dim > stem_out_feats
        # for all of the models we've used so far
        if config['HEAD_DWSCONV']:
            point_conv = torch.nn.Conv2d(stem_out_feats, fc_dim, kernel_size=1, stride=1)
            depth_conv = torch.nn.Conv2d(fc_dim, fc_dim, kernel_size=2, groups=fc_dim)
            self._init_weights(point_conv)
            self._init_weights(depth_conv)
            self.dwsconv = torch.nn.Sequential(point_conv, depth_conv)
            self.classifier = torch.nn.Linear(fc_dim, num_classes)
            ln_feats = fc_dim
        else:
            self.classifier = torch.nn.Linear(stem_out_feats, num_classes)
            ln_feats = stem_out_feats
            
        # convnext needs a layernorm after the stem
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
        x = self.stem(x)
        
        # the structure of the classification head depends on the pretrained stem
        # (eg dropout or normalization might be necessary)
        if self.mode == 'resnet' or self.mode == 'lambda':
            # experiment with dropout layer
            x = self.dropout(x)
            x = self.dwsconv(x)
            # experiment with batchnorm layer
            x = self.bn(x)
            x = self.pool(x)
            x = self.classifier(x)
        if self.mode == 'resnext':
            # experiment with dropout layer
            x = self.dropout(x)
            x = self.dwsconv(x)
            # experiment with batchnorm layer
            x = self.bn(x)
            x = self.pool(x)
            x = self.classifier(x)
        elif self.mode =='efficientnet':
            x = self.bn1(x)
            x = self.blocks(x)
            x = self.bn2(x)
            # experiment with dropout layer
            x = self.dropout(x)
            x = self.dwsconv(x)
            x = self.bn3(x)
            x = self.pool(x)
            x = self.classifier(x)
        elif self.mode == 'convnext':
            x = self.stages(x)
            # experiment with dropout layer
            x = self.dropout(x)
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
        parser.add_argument("--fc_dim", type=int, default=config['FC_DIM'])
        parser.add_argument("--fc_dropout", type=float, default=config['FC_DROPOUT'])
        parser.add_argument("--pretrained_model", type=str, default=config["pretrained_stem"])
        return parser