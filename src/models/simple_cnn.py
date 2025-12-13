import torch
from typing import Optional
from torch import nn
from typing import Literal, List

from src.models.blocks import Blocks

class SimpleCNN(nn.Module):
    def __init__(self, 
                 num_layers:int=5, # m
                 filter_size:int=3, # k
                 num_dense:int=128, # n
                 conv_activation:Literal['relu', 'gelu', 'silu', 'mish']='relu',
                 dense_activation:Literal['relu', 'gelu', 'silu', 'mish']='relu',
                 in_channels:int=3,
                 num_classes:int=6,
                 padding:int|str='same',
                 stride:int=1,
                 dropout:Optional[float]=None,
                 input_size:int=128,
                 base_features:int=16,
                 pooling_k:List[int]|int|None=2,
                 include_batchnorm:bool=True,
                 strategy:Literal['same', 'doubling', 'halving']='same',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        blocks = Blocks()
        out_features = blocks.build_strategy(strategy=strategy,
                                             num_layers=num_layers,
                                             base_features=base_features)
        layers = []
        w, h = input_size, input_size
        pooling_ki=None
        for idx, out_ch in enumerate(out_features):
            if isinstance(pooling_k, list):
                pooling_ki = pooling_k[idx]
            elif isinstance(pooling_k, int):
                pooling_ki = pooling_k
            layers.append(
                blocks.get_block(
                    in_channels=in_channels,
                    out_channels=out_ch,
                    kernel=filter_size,
                    stride=stride,
                    padding=padding,
                    dropout=dropout,
                    include_batchnorm=include_batchnorm,
                    activation=conv_activation,
                    pooling_k=pooling_k if not isinstance(pooling_k, list) else pooling_ki
                )
            )
            in_channels = out_ch
            if isinstance(padding, int):
                w = (w - filter_size + 2 * padding) // stride + 1
                h = (h - filter_size + 2 * padding) // stride + 1
            if pooling_ki:
                w = w // pooling_ki
                h = h // pooling_ki
        
        layers.append(nn.Flatten())
        layers.append(
            blocks.get_block(
                in_channels=in_channels*w*h,
                layer_type='dense',
                out_channels=num_dense,
                activation=dense_activation
            )
        )
        layers.append(
            blocks.get_block(
                in_channels=num_dense,
                out_channels=num_classes,
                layer_type='dense'
            )
        )

        self.model = nn.Sequential(*layers)

    
    def forward(self, x):
        return self.model(x)
    