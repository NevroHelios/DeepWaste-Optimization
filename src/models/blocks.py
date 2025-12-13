import torch
import torch.nn as nn
from typing import Literal, Optional


class Blocks:
    def __init__(self) -> None:
        self.activations = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'mish': nn.Mish,
        }
    
    def build_strategy(self, strategy:Literal['same', 'doubling', 'halving'],
                     num_layers:int,
                     base_features:int):
        """the out features for each block"""
        blocks = []
        if strategy == 'same':
            for _ in range(num_layers):
                blocks.append(base_features)
        elif strategy == 'doubling':
            for i in range(num_layers):
                blocks.append(base_features * (2 ** i))
        elif strategy == 'halving':
            for i in range(num_layers):
                blocks.append(base_features // (2 ** i))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return blocks

    def get_block(self, 
                  in_channels:int,
                  out_channels:int,
                  kernel:int=3,
                  stride:int=1,
                  padding:int|str='same',
                  pooling_k:Optional[int]=None,
                  dropout:Optional[float]=None,
                  include_batchnorm:bool=False,
                  layer_type:Literal['conv', 'dense']='conv',
                  activation:Optional[Literal['relu', 'gelu', 'silu', 'mish']]=None,
                  ):
        block = []
        if layer_type == 'conv':
            block.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )

        elif layer_type == 'dense':
            block.append(
                nn.Linear(
                    in_features=in_channels,
                    out_features=out_channels,
                )
            )
        
        if activation:
            block.append(self.activations[activation]())
        
        if include_batchnorm:
            if layer_type == 'conv':
                block.append(nn.BatchNorm2d(out_channels))
            elif layer_type == 'dense':
                block.append(nn.BatchNorm1d(out_channels))
        
        if pooling_k:
            block.append(nn.MaxPool2d(pooling_k))
    
        if dropout:
            block.append(nn.Dropout(dropout))
        
        return nn.Sequential(*block)