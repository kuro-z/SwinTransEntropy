import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.normalisation import channel, instance
from src.network.swin import SwinTransformer
from torchsummary import summary

class Encoder(nn.Module):
    def __init__(self, C=220):
        """ 
        Encoder with convolutional architecture proposed in [1].
        Projects image x ([C_in,256,256]) into a feature map of size C x W/16 x H/16
        ========
        Arguments:
        image_dims:  Dimensions of input image, (C_in,H,W)
        batch_size:  Number of instances per minibatch
        C:           Bottleneck depth, controls bits-per-pixel
                     C = {2,4,8,16}

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        
        super(Encoder, self).__init__()
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        kernel_dim = 3
        N = 384
        self.n_downsampling_layers = 4
        self.post_pad = nn.ReflectionPad2d(1)

        self.conv_block_out = nn.Sequential(
            self.post_pad,
            nn.Conv2d(N, C, kernel_dim, stride=1),
        )

        self.en = SwinTransformer(in_chans=3,
                            patch_size=2,
                            window_size=4,
                            embed_dim=48,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24))


    def forward(self, x):
        x = self.en(x)
        out = self.conv_block_out(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, input_dims, kernel_size=3, stride=1,
                 channel_norm=True, activation='relu'):
        """
        input_dims: Dimension of input tensor (B,C,H,W)
        """
        super(ResidualBlock, self).__init__()

        self.activation = getattr(F, activation)
        in_channels = input_dims[1]
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)

        if channel_norm is True:
            self.interlayer_norm = channel.ChannelNorm2D_wrap
        else:
            self.interlayer_norm = instance.InstanceNorm2D_wrap

        pad_size = int((kernel_size-1)/2)
        self.pad = nn.ReflectionPad2d(pad_size)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.norm1 = self.interlayer_norm(in_channels, **norm_kwargs)
        self.norm2 = self.interlayer_norm(in_channels, **norm_kwargs)

    def forward(self, x):
        identity_map = x
        res = self.pad(x)
        res = self.conv1(res)
        res = self.norm1(res)
        res = self.activation(res)

        res = self.pad(res)
        res = self.conv2(res)
        res = self.norm2(res)

        return torch.add(res, identity_map)