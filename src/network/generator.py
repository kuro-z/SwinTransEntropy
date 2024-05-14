import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.network.swin import SwinTransformerUp

# Custom
from src.normalisation import channel, instance
from torchsummary import summary

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

class Generator(nn.Module):
    def __init__(self, input_dims, batch_size, C=16, activation='relu',
                 n_residual_blocks=8, channel_norm=True, sample_noise=True,
                 noise_dim=32):

        """ 
        Generator with convolutional architecture proposed in [1].
        Upscales quantized encoder output into feature map of size C x W x H.
        Expects input size (C,16,16)
        ========
        Arguments:
        input_dims: Dimensions of quantized representation, (C,H,W)
        batch_size: Number of instances per minibatch
        C:          Encoder bottleneck depth, controls bits-per-pixel
                    C = 220 used in [1].

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        
        super(Generator, self).__init__()
        
        kernel_dim = 3
        filters = [384]
        self.n_residual_blocks = n_residual_blocks
        self.sample_noise = sample_noise
        self.noise_dim = noise_dim

        # Layer / normalization options
        cnn_kwargs = dict(stride=2, padding=1, output_padding=1)
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu)
        self.n_upsampling_layers = 4
        
        if channel_norm is True:
            self.interlayer_norm = channel.ChannelNorm2D_wrap
        else:
            self.interlayer_norm = instance.InstanceNorm2D_wrap

        self.pre_pad = nn.ReflectionPad2d(1)
        self.asymmetric_pad = nn.ReflectionPad2d((0,1,1,0))  # Slower than tensorflow?
        self.post_pad = nn.ReflectionPad2d(3)

        H0, W0 = input_dims[1:]
        heights = [2**i for i in range(5,9)]
        widths = heights
        H1, H2, H3, H4 = heights
        W1, W2, W3, W4 = widths 


        # (16,16) -> (16,16), with implicit padding
        self.conv_block_init = nn.Sequential(
            self.interlayer_norm(C, **norm_kwargs),
            self.pre_pad,
            nn.Conv2d(C, filters[0], kernel_size=(3,3), stride=1),
            self.interlayer_norm(filters[0], **norm_kwargs),
        )

        if sample_noise is True:
            # Concat noise with latent representation
            filters[0] += self.noise_dim


        for m in range(n_residual_blocks):
            resblock_m = ResidualBlock(input_dims=(batch_size, filters[0], H0, W0),
                channel_norm=channel_norm, activation=activation)
            self.add_module(f'resblock_{str(m)}', resblock_m)
        
        # self.de = SwinTransformerUp(in_chans=3,
        #                           patch_size=2,
        #                           window_size=4,
        #                           embed_dim=48,
        #                           depths=(2, 2, 6, 2),
        #                           num_heads=(3, 6, 12, 24))
        self.de = SwinTransformerUp(in_chans=3,
                                  patch_size=2,
                                  window_size=4,
                                  embed_dim=48,
                                  depths=(2, 6, 2, 2),
                                  num_heads=(24, 12, 6, 3))


    def forward(self, x):
        
        head = self.conv_block_init(x)

        if self.sample_noise is True:
            B, C, H, W = tuple(head.size())
            z = torch.randn((B, self.noise_dim, H, W)).to(head)
            head = torch.cat((head,z), dim=1)

        for m in range(self.n_residual_blocks):
            resblock_m = getattr(self, f'resblock_{str(m)}')
            if m == 0:
                x = resblock_m(head)
            else:
                x = resblock_m(x)

        x += head
        # x = self.upconv_block1(x)
        # x = self.upconv_block2(x)
        # x = self.upconv_block3(x)
        # x = self.upconv_block4(x)
        x = self.de(x)

        return x



if __name__ == "__main__":

    C = 8
    y = torch.randn([3,C,16,16])
    y_dims = y.size()
    G = Generator(y_dims[1:], y_dims[0], C=C, n_residual_blocks=2, sample_noise=False)

    # x_hat = G(y)
    # print(x_hat.size())
    summary(G, input_size=(C, 16, 16), batch_size=3, device="cpu")