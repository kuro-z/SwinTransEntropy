import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

# Custom
from src.network import hyper
from src.helpers import maths, utils
from src.compression import hyperprior_model, prior_model
from src.network.CheckerboardContext import CheckerboardContext
from src.network.CheckerboardContext import CheckerContext

MIN_SCALE = 0.11
LOG_SCALES_MIN = -3.
MIN_LIKELIHOOD = 1e-9
MAX_LIKELIHOOD = 1e3
SMALL_HYPERLATENT_FILTERS = 192
LARGE_HYPERLATENT_FILTERS = 192

HyperInfo = namedtuple(
    "HyperInfo",
    "decoded "
    "latent_nbpp hyperlatent_nbpp total_nbpp latent_qbpp hyperlatent_qbpp total_qbpp",
)

CompressionOutput = namedtuple("CompressionOutput",
   ["hyperlatents_encoded",
    "latents_encoded_a",
    "latents_encoded_b",
    "latents_encoded_c",
    "latents_encoded_d",
    "hyperlatent_spatial_shape",
    "latent_spatial_shape",
    "batch_shape",
    "spatial_shape",
    "hyper_coding_shape",
    "latent_coding_shape_a",
    "latent_coding_shape_b",
    "latent_coding_shape_c",
    "latent_coding_shape_d",
    "hyperlatent_bits",
    "latent_bits",
    "total_bits",
    "hyperlatent_bpp",
    "latent_bpp",
    "total_bpp"]
)

lower_bound_identity = maths.LowerBoundIdentity.apply
lower_bound_toward = maths.LowerBoundToward.apply

class CodingModel(nn.Module):
    """
    Probability model for estimation of (cross)-entropies in the context
    of data compression. TODO: Add tensor -> string compression and
    decompression functionality.
    """

    def __init__(self, n_channels, min_likelihood=MIN_LIKELIHOOD, max_likelihood=MAX_LIKELIHOOD):
        super(CodingModel, self).__init__()
        self.n_channels = n_channels
        self.min_likelihood = float(min_likelihood)
        self.max_likelihood = float(max_likelihood)

    def _quantize(self, x, mode='noise', means=None):
        """
        mode:       If 'noise', returns continuous relaxation of hard
                    quantization through additive uniform noise channel.
                    Otherwise perform actual quantization (through rounding).
        """

        if mode == 'noise':
            quantization_noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
            x = x + quantization_noise

        elif mode == 'quantize':
            if means is not None:
                x = x - means
                x = torch.floor(x + 0.5)
                x = x + means
            else:
                x = torch.floor(x + 0.5)
        else:
            raise NotImplementedError
        
        return x

    def _estimate_entropy(self, likelihood, spatial_shape):

        EPS = 1e-9  
        quotient = -np.log(2.)
        batch_size = likelihood.size()[0]

        assert len(spatial_shape) == 2, 'Mispecified spatial dims'
        n_pixels = np.prod(spatial_shape)

        log_likelihood = torch.log(likelihood + EPS)
        n_bits = torch.sum(log_likelihood) / (batch_size * quotient)
        bpp = n_bits / n_pixels

        return n_bits, bpp

    def _estimate_entropy_log(self, log_likelihood, spatial_shape):

        quotient = -np.log(2.)
        batch_size = log_likelihood.size()[0]

        assert len(spatial_shape) == 2, 'Mispecified spatial dims'
        n_pixels = np.prod(spatial_shape)

        n_bits = torch.sum(log_likelihood) / (batch_size * quotient)
        bpp = n_bits / n_pixels

        return n_bits, bpp

    def quantize_latents_st(self, inputs, means=None):
        # Latents rounded instead of additive uniform noise
        # Ignore rounding in backward pass
        values = inputs

        if means is not None:
            values = values - means

        delta = (torch.floor(values + 0.5) - values).detach()
        values = values + delta

        if means is not None:
            values = values + means

        return values

    def latent_likelihood(self, x, mean, scale):

        # Assumes 1 - CDF(x) = CDF(-x)
        x = x - mean
        x = torch.abs(x)
        cdf_upper = self.standardized_CDF((0.5 - x) / scale)
        cdf_lower = self.standardized_CDF(-(0.5 + x) / scale)

        # Naive
        # cdf_upper = self.standardized_CDF( (x + 0.5) / scale )
        # cdf_lower = self.standardized_CDF( (x - 0.5) / scale )

        likelihood_ = cdf_upper - cdf_lower
        likelihood_ = lower_bound_toward(likelihood_, self.min_likelihood)

        return likelihood_


class Hyperprior(CodingModel):
    
    def __init__(self, bottleneck_capacity=220, hyperlatent_filters=LARGE_HYPERLATENT_FILTERS,
        mode='large', likelihood_type='gaussian', scale_lower_bound=MIN_SCALE, entropy_code=False,
        vectorize_encoding=True, block_encode=True):

        """
        Introduces probabilistic model over latents of 
        latents.

        The hyperprior over the standard latents is modelled as
        a non-parametric, fully factorized density.
        """

        super(Hyperprior, self).__init__(n_channels=bottleneck_capacity)
        
        self.bottleneck_capacity = bottleneck_capacity
        self.scale_lower_bound = scale_lower_bound

        analysis_net = hyper.HyperpriorAnalysis
        synthesis_net = hyper.HyperpriorSynthesis
        synthesis_net_1 = hyper.HyperpriorSynthesis_1
        # synthesis_net2 = hyper.HyperpriorSynthesis_2()
        # synthesis_net1 = hyper.HyperpriorSynthesis
        self.M = hyperlatent_filters

        if mode == 'small':
            hyperlatent_filters = SMALL_HYPERLATENT_FILTERS

        self.analysis_net = analysis_net(C=bottleneck_capacity, N=hyperlatent_filters)
        self.synthesis_net = synthesis_net(C=hyperlatent_filters * 2, N=hyperlatent_filters)

        # self.synthesis_mu = synthesis_net(C=bottleneck_capacity, N=hyperlatent_filters)
        # self.synthesis_std = synthesis_net(C=bottleneck_capacity, N=hyperlatent_filters)

        # self.synthesis_head = synthesis_net2(C=hyperlatent_filters * 2, N=hyperlatent_filters * 4)
        self.synthesis_mu = synthesis_net_1(C=bottleneck_capacity, N=hyperlatent_filters * 4)
        self.synthesis_std = synthesis_net_1(C=bottleneck_capacity, N=hyperlatent_filters * 4)

        # self.context_prediction = CheckerboardContext(bottleneck_capacity, hyperlatent_filters*2, 5, 1, 2)
        self.context_prediction = CheckerContext(C=bottleneck_capacity, N=hyperlatent_filters * 2)
        
        self.amortization_models = [self.analysis_net, self.synthesis_net, self.context_prediction, self.synthesis_mu, self.synthesis_std]

        self.hyperlatent_likelihood = hyperprior_model.HyperpriorDensity(n_channels=hyperlatent_filters)

        if likelihood_type == 'gaussian':
            self.standardized_CDF = maths.standardized_CDF_gaussian
        elif likelihood_type == 'logistic':
            self.standardized_CDF = maths.standardized_CDF_logistic
        else:
            raise ValueError('Unknown likelihood model: {}'.format(likelihood_type))

        if entropy_code is True:
            print('Building prior probability tables...')
            self.hyperprior_entropy_model = hyperprior_model.HyperpriorEntropyModel(
                distribution=self.hyperlatent_likelihood)
            self.prior_density = prior_model.PriorDensity(n_channels=bottleneck_capacity,
                scale_lower_bound=self.scale_lower_bound, likelihood_type=likelihood_type)
            self.prior_entropy_model = prior_model.PriorEntropyModel(
                distribution=self.prior_density, min_scale=self.scale_lower_bound)
            self.index_tables = self.prior_entropy_model.scale_table_tensor
            self.vectorize_encoding = vectorize_encoding
            self.block_encode = block_encode

    def compress_forward(self, latents, spatial_shape, **kwargs):
        latent_spatial_shape = latents.shape[2:]
        latents_height, latents_width = latents.shape[2:]

        latents_a = latents[:, :, 0::2, 0::2]
        latents_d = latents[:, :, 1::2, 1::2]
        latents_b = latents[:, :, 0::2, 1::2]
        latents_c = latents[:, :, 1::2, 0::2]

        # Obtain hyperlatents from hyperencoder
        hyperlatents = self.analysis_net(latents)
        hyperlatent_spatial_shape = hyperlatents.size()[2:]
        batch_shape = latents.size(0)

        # Estimate Shannon entropies for hyperlatents
        hyp_agg = self.hyperprior_entropy_model._estimate_compression_bits(
            hyperlatents, spatial_shape)
        hyperlatent_bits, hyperlatent_bpp, hyperlatent_bpi = hyp_agg

        # Compress, then decompress hyperlatents
        hyperlatents_encoded, hyper_coding_shape, _ = self.hyperprior_entropy_model.compress(hyperlatents,
            vectorize=self.vectorize_encoding, block_encode=self.block_encode)
        hyperlatents_decoded, _ = self.hyperprior_entropy_model.decompress(hyperlatents_encoded,
            batch_shape=batch_shape, broadcast_shape=hyperlatent_spatial_shape,
            coding_shape=hyper_coding_shape, vectorize=self.vectorize_encoding, block_decode=self.block_encode)
        hyperlatents_decoded = hyperlatents_decoded.to(latents)

        latent_syn = self.synthesis_net(hyperlatents_decoded)
        anchor = torch.zeros_like(latents).to(latents.device)
        anchor[:, :, 0::2, 1::2] = latents[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = latents[:, :, 1::2, 0::2]
        ctx_params_anchor = torch.zeros([batch_shape, self.M * 2, latents_height, latents_width]).to(latents.device)
        latent_anchor = torch.cat([ctx_params_anchor, latent_syn], dim=1)
        # latent_anchor_head = self.synthesis_head(latent_anchor)
        latent_means_anchor = self.synthesis_mu(latent_anchor)
        latent_scales_anchor = self.synthesis_std(latent_anchor)

        #----------b--------------#
        latent_spatial_shape_b = latents_b.shape[2:]
        latent_scales_b = latent_scales_anchor[:, :, 0::2, 1::2]
        latent_means_b = latent_means_anchor[:, :, 0::2, 1::2]
        latent_scales_indexes_b = lower_bound_toward(latent_scales_b, self.scale_lower_bound)
        # Use latent statistics to build indexed probability tables, and compress latents
        latents_encoded_b, latent_coding_shape_b, _ = self.prior_entropy_model.compress(latents_b, means=latent_means_b,
            scales=latent_scales_indexes_b, vectorize=self.vectorize_encoding, block_encode=self.block_encode)
        # Estimate Shannon entropies for latents
        latent_agg_b = self.prior_entropy_model._estimate_compression_bits(latents_b,
            means=latent_means_b, scales=latent_scales_indexes_b, spatial_shape=spatial_shape)
        latents_decoded_b, _ = self.prior_entropy_model.decompress(latents_encoded_b, means=latent_means_b,
            scales=latent_scales_indexes_b, broadcast_shape=latent_spatial_shape_b, coding_shape=latent_coding_shape_b,
            vectorize=self.vectorize_encoding,block_decode=self.block_encode)
        latent_bits_b, latent_bpp_b, latent_bpi_b = latent_agg_b

        #------------c-----------------#
        latent_spatial_shape_c = latents_c.shape[2:]
        latent_scales_c = latent_scales_anchor[:, :, 1::2, 0::2]
        latent_means_c = latent_means_anchor[:, :, 1::2, 0::2]
        latent_scales_indexes_c = lower_bound_toward(latent_scales_c, self.scale_lower_bound)
        latents_encoded_c, latent_coding_shape_c, _ = self.prior_entropy_model.compress(latents_c, means=latent_means_c,
            scales=latent_scales_indexes_c, vectorize=self.vectorize_encoding, block_encode=self.block_encode)
        # Estimate Shannon entropies for latents
        latent_agg_c = self.prior_entropy_model._estimate_compression_bits(latents_c,
             means=latent_means_c, scales=latent_scales_indexes_c, spatial_shape=spatial_shape)

        latents_decoded_c, _ = self.prior_entropy_model.decompress(latents_encoded_c, means=latent_means_c,
             scales=latent_scales_indexes_c, broadcast_shape=latent_spatial_shape_c, coding_shape=latent_coding_shape_c,
             vectorize=self.vectorize_encoding, block_decode=self.block_encode)
        latent_bits_c, latent_bpp_c, latent_bpi_c = latent_agg_c


        anchor_quantized = torch.zeros_like(latents).to(latents.device)
        anchor_quantized[:, :, 0::2, 1::2] = latents_decoded_b[:, :, :, :]
        anchor_quantized[:, :, 1::2, 0::2] = latents_decoded_c[:, :, :, :]
        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        latent_non_anchor = torch.cat([ctx_params_non_anchor, latent_syn], dim=1)
        # latent_non_anchor_head = self.synthesis_head(latent_non_anchor)
        latent_means_non_anchor = self.synthesis_mu(latent_non_anchor)
        latent_scales_non_anchor = self.synthesis_std(latent_non_anchor)

        # ---------------a-----------------#
        latent_scales_a = latent_scales_non_anchor[:, :, 0::2, 0::2]
        latent_means_a = latent_means_non_anchor[:, :, 0::2, 0::2]
        latent_scales_indexes_a = lower_bound_toward(latent_scales_a, self.scale_lower_bound)
        latents_encoded_a, latent_coding_shape_a, _ = self.prior_entropy_model.compress(latents_a, means=latent_means_a,
             scales=latent_scales_indexes_a, vectorize=self.vectorize_encoding, block_encode=self.block_encode)
        # Estimate Shannon entropies for latents
        latent_agg_a = self.prior_entropy_model._estimate_compression_bits(latents_a,
             means=latent_means_a, scales=latent_scales_indexes_a, spatial_shape=spatial_shape)
        latent_bits_a, latent_bpp_a, latent_bpi_a = latent_agg_a

        # ---------------d-----------------#
        latent_scales_d = latent_scales_non_anchor[:, :, 1::2, 1::2]
        latent_means_d = latent_means_non_anchor[:, :, 1::2, 1::2]
        latent_scales_indexes_d = lower_bound_toward(latent_scales_d, self.scale_lower_bound)
        latents_encoded_d, latent_coding_shape_d, _ = self.prior_entropy_model.compress(latents_d, means=latent_means_d,
              scales=latent_scales_indexes_d, vectorize=self.vectorize_encoding, block_encode=self.block_encode)
        # Estimate Shannon entropies for latents
        latent_agg_d = self.prior_entropy_model._estimate_compression_bits(latents_d, means=latent_means_d,
              scales=latent_scales_indexes_d, spatial_shape=spatial_shape)
        latent_bits_d, latent_bpp_d, latent_bpi_d = latent_agg_d

        latent_bits = latent_bits_a + latent_bits_b + latent_bits_c + latent_bits_d
        latent_bpp = latent_bpp_a + latent_bpp_d + latent_bpp_c + latent_bpp_d


        # What the decoder needs for reconstruction
        compression_output = CompressionOutput(
            hyperlatents_encoded=hyperlatents_encoded,
            latents_encoded_a=latents_encoded_a,
            latents_encoded_b=latents_encoded_b,
            latents_encoded_c=latents_encoded_c,
            latents_encoded_d=latents_encoded_d,
            hyperlatent_spatial_shape=hyperlatent_spatial_shape,  # 2D
            latent_spatial_shape=latent_spatial_shape,
            spatial_shape=spatial_shape,  # 2D
            hyper_coding_shape=hyper_coding_shape,  # C,H,W
            latent_coding_shape_a=latent_coding_shape_a,  # C,H,W
            latent_coding_shape_b=latent_coding_shape_b,  # C,H,W
            latent_coding_shape_c=latent_coding_shape_c,  # C,H,W
            latent_coding_shape_d=latent_coding_shape_d,  # C,H,W
            batch_shape=batch_shape,
            hyperlatent_bits=hyperlatent_bits.item(),  # for reporting 
            latent_bits=latent_bits.item(),
            total_bits=(hyperlatent_bits + latent_bits).item(),
            hyperlatent_bpp=hyperlatent_bpp.item(),
            latent_bpp=latent_bpp.item(),
            total_bpp=(hyperlatent_bpp + latent_bpp).item(),
        )

        return compression_output

    def decompress_forward(self, compression_output, device):

        hyperlatents_encoded = compression_output.hyperlatents_encoded
        latents_encoded_a = compression_output.latents_encoded_a
        latents_encoded_b = compression_output.latents_encoded_b
        latents_encoded_c = compression_output.latents_encoded_c
        latents_encoded_d = compression_output.latents_encoded_d
        hyperlatent_spatial_shape = compression_output.hyperlatent_spatial_shape
        latent_spatial_shape = compression_output.latent_spatial_shape
        batch_shape = compression_output.batch_shape

        # Decompress hyperlatents
        hyperlatents_decoded, _ = self.hyperprior_entropy_model.decompress(hyperlatents_encoded,
            batch_shape=batch_shape, broadcast_shape=hyperlatent_spatial_shape,
            coding_shape=compression_output.hyper_coding_shape, vectorize=self.vectorize_encoding,
            block_decode=self.block_encode)
        hyperlatents_decoded = hyperlatents_decoded.to(device)

        latent_syn = self.synthesis_net(hyperlatents_decoded)
        ctx_params_anchor = torch.zeros([batch_shape, self.M * 2, latent_spatial_shape[0], latent_spatial_shape[1]]).to(device)
        latent_anchor = torch.cat([ctx_params_anchor, latent_syn], dim=1)
        # latent_anchor_head = self.synthesis_head(latent_anchor)
        latent_means_anchor = self.synthesis_mu(latent_anchor)
        latent_scales_anchor = self.synthesis_std(latent_anchor)

        #--------------------b----------------------#
        latent_scales_b = latent_scales_anchor[:, :, 0::2, 1::2]
        latent_means_b = latent_means_anchor[:, :, 0::2, 1::2]
        latent_scales_indexes_b = lower_bound_toward(latent_scales_b, self.scale_lower_bound)
        latent_spatial_shape_b = latent_scales_b.size()[2:]
        latents_decoded_b, _ = self.prior_entropy_model.decompress(latents_encoded_b, means=latent_means_b,
             scales=latent_scales_indexes_b, broadcast_shape=latent_spatial_shape_b, coding_shape=compression_output.latent_coding_shape_b,
             vectorize=self.vectorize_encoding, block_decode=self.block_encode)

        # --------------------c----------------------#
        latent_scales_c = latent_scales_anchor[:, :, 1::2, 0::2]
        latent_means_c = latent_means_anchor[:, :, 1::2, 0::2]
        latent_scales_indexes_c = lower_bound_toward(latent_scales_c, self.scale_lower_bound)
        latent_spatial_shape_c = latent_scales_c.size()[2:]
        latents_decoded_c, _ = self.prior_entropy_model.decompress(latents_encoded_c, means=latent_means_c,
             scales=latent_scales_indexes_c, broadcast_shape=latent_spatial_shape_c, coding_shape=compression_output.latent_coding_shape_c,
             vectorize=self.vectorize_encoding,  block_decode=self.block_encode)

        anchor_quantized = torch.zeros([batch_shape, self.M, latent_spatial_shape[0], latent_spatial_shape[1]]).to(device)
        anchor_quantized[:, :, 0::2, 1::2] = latents_decoded_b[:, :, :, :]
        anchor_quantized[:, :, 1::2, 0::2] = latents_decoded_c[:, :, :, :]
        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        latent_non_anchor = torch.cat([ctx_params_non_anchor, latent_syn], dim=1)
        # latent_non_anchor_head = self.synthesis_head(latent_non_anchor)
        latent_means_non_anchor = self.synthesis_mu(latent_non_anchor)
        latent_scales_non_anchor = self.synthesis_std(latent_non_anchor)

        #-------------------a---------------------#
        latent_scales_a = latent_scales_non_anchor[:, :, 0::2, 0::2]
        latent_means_a = latent_means_non_anchor[:, :, 0::2, 0::2]
        latent_scales_indexes_a = lower_bound_toward(latent_scales_a, self.scale_lower_bound)
        latent_spatial_shape_a = latent_scales_a.size()[2:]
        latents_decoded_a, _ = self.prior_entropy_model.decompress(latents_encoded_a, means=latent_means_a,
             scales=latent_scales_indexes_a, broadcast_shape=latent_spatial_shape_a, coding_shape=compression_output.latent_coding_shape_a,
             vectorize=self.vectorize_encoding, block_decode=self.block_encode)

        # -------------------d---------------------#
        latent_scales_d = latent_scales_non_anchor[:, :, 1::2, 1::2]
        latent_means_d = latent_means_non_anchor[:, :, 1::2, 1::2]
        latent_scales_indexes_d = lower_bound_toward(latent_scales_d, self.scale_lower_bound)
        latent_spatial_shape_d = latent_scales_d.size()[2:]
        latents_decoded_d, _ = self.prior_entropy_model.decompress(latents_encoded_d, means=latent_means_d,
             scales=latent_scales_indexes_d, broadcast_shape=latent_spatial_shape_d, coding_shape=compression_output.latent_coding_shape_d,
             vectorize=self.vectorize_encoding, block_decode=self.block_encode)

        anchor_quantized[:, :, 0::2, 0::2] = latents_decoded_a[:, :, :, :]
        anchor_quantized[:, :, 1::2, 1::2] = latents_decoded_d[:, :, :, :]

        return anchor_quantized


    def forward(self, latents, spatial_shape, **kwargs):
        latents_height, latents_width = latents.shape[2:]
        batch_shape = latents.size(0)

        hyperlatents = self.analysis_net(latents)

        # Mismatch b/w continuous and discrete cases?
        # Differential entropy, hyperlatents
        noisy_hyperlatents = self._quantize(hyperlatents, mode='noise')
        noisy_hyperlatent_likelihood = self.hyperlatent_likelihood(noisy_hyperlatents)
        noisy_hyperlatent_bits, noisy_hyperlatent_bpp = self._estimate_entropy(
            noisy_hyperlatent_likelihood, spatial_shape)

        # Discrete entropy, hyperlatents
        quantized_hyperlatents = self._quantize(hyperlatents, mode='quantize')
        quantized_hyperlatent_likelihood = self.hyperlatent_likelihood(quantized_hyperlatents)
        quantized_hyperlatent_bits, quantized_hyperlatent_bpp = self._estimate_entropy(
            quantized_hyperlatent_likelihood, spatial_shape)

        if self.training is True:
            hyperlatents_decoded = noisy_hyperlatents
        else:
            hyperlatents_decoded = quantized_hyperlatents

        latent_syn = self.synthesis_net(hyperlatents_decoded)
        
        #----------------------------------------#
        noisy_latents = self._quantize(latents, mode='noise')
        quantized_latents = self._quantize(latents, mode='quantize')
        if self.training is True:
            latents_decoded = noisy_latents
        else:
            latents_decoded = quantized_latents
        y_half = latents_decoded.clone()
        #----------------------------------------#

        anchor = torch.zeros_like(latents).to(latents.device)
        anchor[:, :, 0::2, 1::2] = y_half[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y_half[:, :, 1::2, 0::2]

        ctx_params_anchor = torch.zeros([batch_shape, self.M * 2, latents_height, latents_width]).to(latents.device)

        latent_anchor = torch.cat([ctx_params_anchor, latent_syn], dim=1)
        # latent_anchor_head = self.synthesis_head(latent_anchor)
        latent_means_anchor = self.synthesis_mu(latent_anchor)
        latent_scales_anchor = self.synthesis_std(latent_anchor)

        ctx_params_non_anchor = self.context_prediction(anchor)
        #-------------------------------#
        ctx_params_non_anchor[:, :, 0::2, 1::2] = 0
        ctx_params_non_anchor[:, :, 1::2, 0::2] = 0
        #-------------------------------#
        latent_non_anchor = torch.cat([ctx_params_non_anchor, latent_syn], dim=1)
        # latent_non_anchor_head = self.synthesis_head(latent_non_anchor)
        latent_means_non_anchor = self.synthesis_mu(latent_non_anchor)
        latent_scales_non_anchor = self.synthesis_std(latent_non_anchor)

        latent_scales = torch.zeros([batch_shape, self.M, latents_height, latents_width]).to(latents.device)
        latent_means = torch.zeros([batch_shape, self.M, latents_height, latents_width]).to(latents.device)
        latent_scales[:, :, 0::2, 1::2] = latent_scales_anchor[:, :, 0::2, 1::2]
        latent_scales[:, :, 1::2, 0::2] = latent_scales_anchor[:, :, 1::2, 0::2]
        latent_scales[:, :, 0::2, 0::2] = latent_scales_non_anchor[:, :, 0::2, 0::2]
        latent_scales[:, :, 1::2, 1::2] = latent_scales_non_anchor[:, :, 1::2, 1::2]

        latent_means[:, :, 0::2, 1::2] = latent_means_anchor[:, :, 0::2, 1::2]
        latent_means[:, :, 1::2, 0::2] = latent_means_anchor[:, :, 1::2, 0::2]
        latent_means[:, :, 0::2, 0::2] = latent_means_non_anchor[:, :, 0::2, 0::2]
        latent_means[:, :, 1::2, 1::2] = latent_means_non_anchor[:, :, 1::2, 1::2]

        latent_scales = lower_bound_toward(latent_scales, self.scale_lower_bound)

        # Differential entropy, latents
        # noisy_latents = self._quantize(latents, mode='noise', means=latent_means)
        noisy_latent_likelihood = self.latent_likelihood(noisy_latents, mean=latent_means,
            scale=latent_scales)
        noisy_latent_bits, noisy_latent_bpp = self._estimate_entropy(
            noisy_latent_likelihood, spatial_shape)

        # Discrete entropy, latents
        # quantized_latents = self._quantize(latents, mode='quantize', means=latent_means)
        quantized_latent_likelihood = self.latent_likelihood(quantized_latents, mean=latent_means,
            scale=latent_scales)
        quantized_latent_bits, quantized_latent_bpp = self._estimate_entropy(
            quantized_latent_likelihood, spatial_shape)

        # latents_decoded = self.quantize_latents_st(latents, latent_means)

        info = HyperInfo(
            decoded=latents_decoded,
            latent_nbpp=noisy_latent_bpp,
            hyperlatent_nbpp=noisy_hyperlatent_bpp,
            total_nbpp=noisy_latent_bpp + noisy_hyperlatent_bpp,
            latent_qbpp=quantized_latent_bpp,
            hyperlatent_qbpp=quantized_hyperlatent_bpp,
            total_qbpp=quantized_latent_bpp + quantized_hyperlatent_bpp,
        )

        return info


"""
========
Discretized logistic mixture model.
========
"""


class HyperpriorDLMM(CodingModel):
    
    def __init__(self, bottleneck_capacity=64, hyperlatent_filters=LARGE_HYPERLATENT_FILTERS, mode='large',
        likelihood_type='gaussian', scale_lower_bound=MIN_SCALE, mixture_components=4, 
        entropy_code=False):
        """
        Introduces probabilistic model over latents of 
        latents.

        The hyperprior over the standard latents is modelled as
        a non-parametric, fully factorized density.
        """
        super(HyperpriorDLMM, self).__init__(n_channels=bottleneck_capacity)
        
        assert bottleneck_capacity <= 128, 'Will probably run out of memory!'
        self.bottleneck_capacity = bottleneck_capacity
        self.scale_lower_bound = scale_lower_bound
        self.mixture_components = mixture_components

        analysis_net = hyper.HyperpriorAnalysis
        synthesis_net = hyper.HyperpriorSynthesisDLMM

        if mode == 'small':
            hyperlatent_filters = SMALL_HYPERLATENT_FILTERS

        self.analysis_net = analysis_net(C=bottleneck_capacity, N=hyperlatent_filters)

        # TODO: Combine scale, loc into single network
        self.synthesis_DLMM_params = synthesis_net(C=bottleneck_capacity, N=hyperlatent_filters)
    
        self.amortization_models = [self.analysis_net, self.synthesis_DLMM_params]

        self.hyperlatent_likelihood = hyperprior_model.HyperpriorDensity(n_channels=hyperlatent_filters)

        if likelihood_type == 'gaussian':
            self.standardized_CDF = maths.standardized_CDF_gaussian
        elif likelihood_type == 'logistic':
            self.standardized_CDF = maths.standardized_CDF_logistic
        else:
            raise ValueError('Unknown likelihood model: {}'.format(likelihood_type))

    def latent_log_likelihood_DLMM(self, x, DLMM_params):

        # (B C K H W)
        x, (logit_pis, means, log_scales), K = hyper.unpack_likelihood_params(x, 
            DLMM_params, LOG_SCALES_MIN)

        # Assumes 1 - CDF(x) = CDF(-x) symmetry
        # Numerical stability, do subtraction in left tail

        x_centered = x - means
        x_centered = torch.abs(x_centered)
        inv_stds = torch.exp(-log_scales)
        cdf_upper = self.standardized_CDF(inv_stds * (0.5 - x_centered))
        cdf_lower = self.standardized_CDF(inv_stds * (- 0.5 - x_centered))
        pmf_mixture_component = lower_bound_toward(cdf_upper - cdf_lower, MIN_LIKELIHOOD)
        log_pmf_mixture_component = torch.log(pmf_mixture_component)

        # Non-negativity + normalization via softmax
        lse_in = F.log_softmax(logit_pis, dim=2) + log_pmf_mixture_component
        log_DLMM = torch.logsumexp(lse_in, dim=2)

        return log_DLMM

    def forward(self, latents, spatial_shape, **kwargs):

        hyperlatents = self.analysis_net(latents)
        
        # Mismatch b/w continuous and discrete cases?
        # Differential entropy, hyperlatents
        noisy_hyperlatents = self._quantize(hyperlatents, mode='noise')
        noisy_hyperlatent_likelihood = self.hyperlatent_likelihood(noisy_hyperlatents)
        noisy_hyperlatent_bits, noisy_hyperlatent_bpp = self._estimate_entropy(
            noisy_hyperlatent_likelihood, spatial_shape)

        # Discrete entropy, hyperlatents
        quantized_hyperlatents = self._quantize(hyperlatents, mode='quantize')
        quantized_hyperlatent_likelihood = self.hyperlatent_likelihood(quantized_hyperlatents)
        quantized_hyperlatent_bits, quantized_hyperlatent_bpp = self._estimate_entropy(
            quantized_hyperlatent_likelihood, spatial_shape)

        if self.training is True:
            hyperlatents_decoded = noisy_hyperlatents
        else:
            hyperlatents_decoded = quantized_hyperlatents

        latent_DLMM_params = self.synthesis_DLMM_params(hyperlatents_decoded)

        # Differential entropy, latents
        noisy_latents = self._quantize(latents, mode='noise')
        noisy_latent_log_likelihood = self.latent_log_likelihood_DLMM(noisy_latents, 
            DLMM_params=latent_DLMM_params)
        noisy_latent_bits, noisy_latent_bpp = self._estimate_entropy(
            noisy_latent_log_likelihood, spatial_shape)     

        # Discrete entropy, latents
        quantized_latents = self._quantize(latents, mode='quantize')
        quantized_latent_log_likelihood = self.latent_log_likelihood_DLMM(quantized_latents, 
            DLMM_params=latent_DLMM_params)
        quantized_latent_bits, quantized_latent_bpp = self._estimate_entropy(
            quantized_latent_log_likelihood, spatial_shape)     


        if self.training is True:
            latents_decoded = self.quantize_latents_st(latents)
        else:
            latents_decoded = quantized_latents

        info = HyperInfo(
            decoded=latents_decoded,
            latent_nbpp=noisy_latent_bpp,
            hyperlatent_nbpp=noisy_hyperlatent_bpp,
            total_nbpp=noisy_latent_bpp + noisy_hyperlatent_bpp,
            latent_qbpp=quantized_latent_bpp,
            hyperlatent_qbpp=quantized_hyperlatent_bpp,
            total_qbpp=quantized_latent_bpp + quantized_hyperlatent_bpp,
        )

        return info

        

if __name__ == '__main__':

    def pad_factor(input_image, spatial_dims, factor):
        """Pad `input_image` (N,C,H,W) such that H and W are divisible by `factor`."""
        H, W = spatial_dims[0], spatial_dims[1]
        pad_H = (factor - (H % factor)) % factor
        pad_W = (factor - (W % factor)) % factor
        return F.pad(input_image, pad=(0, pad_W, 0, pad_H), mode='reflect')

    C = 8
    hp = Hyperprior(C)
    hp_dlmm = HyperpriorDLMM(8)

    y = torch.randn((3,C,16,16))
    # y = torch.randn((10,C,126,95))

    n_downsamples = hp.analysis_net.n_downsampling_layers
    factor = 2 ** n_downsamples
    print('Padding to {}'.format(factor))
    y = pad_factor(y, y.size()[2:], factor)
    print('Size after padding', y.size())

    f = hp(y, spatial_shape=(1,1))
    print('Shape of decoded latents', f.decoded.shape)

    f_dlmm = hp_dlmm(y, spatial_shape=(1,1))
    print('Shape of decoded latents', f_dlmm.decoded.shape)
