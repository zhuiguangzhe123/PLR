# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from torch import Tensor
from tqdm import tqdm
import os

try:
    from range_coder import RangeEncoder, RangeDecoder
except:
    pass

from compressai.entropy_models import EntropyBottleneck, GaussianMixtureConditional
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    GaussianConditionalLatentCodec_ST,
    PolynomialLaplaceConditionalLatentCodec_ST,
    HyperLatentCodec,
    HyperpriorLatentCodec,
)
from compressai.layers import (
    AttentionBlock,
    CheckerboardMaskedConv2d,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv1x1,
    conv3x3,
    sequential_channel_ramp,
    subpel_conv3x3,
    MaskedConv2d,
)
from compressai.registry import register_model
from itertools import accumulate

from .base import SimpleVAECompressionModel, CompressionModel
from .utils import conv, deconv

__all__ = [
    "Cheng2020AnchorCheckerboard",
    "Elic2022Official",
    "Elic2022Chandelier",
]


@register_model("cheng2020-anchor-checkerboard")
class Cheng2020AnchorCheckerboard(SimpleVAECompressionModel):
    """Cheng2020 anchor model with checkerboard context model.

    Base transform model from [Cheng2020]. Context model from [He2021].

    [Cheng2020]: `"Learned Image Compression with Discretized Gaussian
    Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun,
    Masaru Takeuchi, and Jiro Katto, CVPR 2020.

    [He2021]: `"Checkerboard Context Model for Efficient Learned Image
    Compression" <https://arxiv.org/abs/2103.15306>`_, by Dailan He,
    Yaoyan Zheng, Baocheng Sun, Yan Wang, and Hongwei Qin, CVPR 2021.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(**kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                "y": CheckerboardLatentCodec(
                    latent_codec={
                        "y": GaussianConditionalLatentCodec(quantizer="ste"),
                    },
                    entropy_parameters=nn.Sequential(
                        nn.Conv2d(N * 12 // 3, N * 10 // 3, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(N * 10 // 3, N * 8 // 3, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(N * 8 // 3, N * 6 // 3, 1),
                    ),
                    context_prediction=CheckerboardMaskedConv2d(
                        N, 2 * N, kernel_size=5, stride=1, padding=2
                    ),
                ),
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


class EntropyParameters(nn.Module):
    def __init__(self, N=192, K=3) -> None:
        super().__init__()

        self.N = N
        self.K = K
        self.layers = nn.Sequential(
            nn.Conv2d(N * 12 // 3, N * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 10 // 3, N * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 10 // 3, 3 * N * K, 1),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class Cheng2020GMM(CompressionModel):
    def __init__(self, N=192, K=3, **kwargs):
        super().__init__(N, **kwargs)

        self.N = N
        self.K = K

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.context_prediction = MaskedConv2d(
            N, 2 * N, kernel_size=5, padding=2, stride=1
        )

        self.entropy_parameters = EntropyParameters(N=N, K=K)
        self.gaussian_conditional = GaussianMixtureConditional()

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        # just round, no need to minus means
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize", means=None
        )

        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat, weights_hat = gaussian_params.chunk(3, 1)

        B, C, H, W = y_hat.shape
        scales_hat = scales_hat.view(B, self.K, C, H, W)
        means_hat = means_hat.view(B, self.K, C, H, W)
        weights_hat = weights_hat.view(B, self.K, C, H, W)
        weights_hat = F.softmax(weights_hat, dim=1)

        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means_hat, weights_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # range-coder is used to coding y_hat
    def compress(self, x, stream_name):
        """
        https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/network.py
        When batch_size is 1
        """
        torch.backends.cudnn.deterministic = True

        # path to save codestreams
        z_stream_path = os.path.join(stream_name + '.npz')
        y_stream_path = os.path.join(stream_name + '.bin')

        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat)

        # just round, no need to minus means
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize", means=None
        )

        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat, weights_hat = gaussian_params.chunk(3, 1)

        B, C, H, W = y_hat.shape
        scales_hat = scales_hat.view(B, self.K, C, H, W)
        means_hat = means_hat.view(B, self.K, C, H, W)
        weights_hat = weights_hat.view(B, self.K, C, H, W)
        weights_hat = F.softmax(weights_hat, dim=1)

        # compute the zero channel and abs'max
        y_hat_np = y_hat.cpu().numpy().astype('int')
        flag = np.zeros(C, dtype=np.int32)
        for ch in range(C):
            if np.sum(abs(y_hat_np[:, ch, :, :])) > 0:
                flag[ch] = 1

        non_zero_idx = np.squeeze(np.where(flag == 1))
        flag_nums = np.packbits(flag.reshape([8, C // 8]))
        minmax = np.maximum(abs(y_hat_np.max()), abs(y_hat_np.min()))
        minmax = int(np.maximum(minmax, 1))

        # write z
        fileobj = open(z_stream_path, mode="wb")
        # x.H, x.W
        fileobj.write(np.array(x.shape[2:], dtype=np.uint16).tobytes())
        fileobj.write(np.array([len(z_strings[0]), minmax], dtype=np.uint16).tobytes())
        fileobj.write(np.array(flag_nums, dtype=np.uint8).tobytes())
        fileobj.write(z_strings[0])
        fileobj.close()

        # compress y_hat
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2
        encoder = RangeEncoder(y_stream_path)
        samples = torch.arange(0, minmax * 2 + 1).float().to(x.device)
        y_hat = F.pad(y_hat, (padding, padding, padding, padding))

        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in tqdm(range(H)):
            for w in range(W):
                y_crop = y_hat[0: 1, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    weight=masked_weight,
                    bias=self.context_prediction.bias
                )
                p = params[0: 1, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_crop, means_crop, weights_crop = gaussian_params.chunk(3, 1)     # (1, 3C, H, W)
                scales_crop = scales_crop.view(self.K, self.N)                          # (1, C, H, W)
                means_crop = means_crop.view(self.K, self.N) + minmax
                weights_crop = weights_crop.view(self.K, self.N)
                weights_crop = F.softmax(weights_crop, dim=0)

                for i in range(len(non_zero_idx)):
                    ch = non_zero_idx[i]
                    scale_p = scales_crop[:, ch]
                    mean_p = means_crop[:, ch]
                    weight_p = weights_crop[:, ch]

                    pmf = torch.zeros_like(samples)
                    for k in range(self.K):
                        half = float(0.5)
                        value = abs(samples - mean_p[k])
                        scale = self.gaussian_conditional.lower_bound_scale(scale_p[k])
                        upper = self.gaussian_conditional._standardized_cumulative((half - value) / scale)
                        lower = self.gaussian_conditional._standardized_cumulative((-half - value) / scale)
                        pmf += (upper - lower) * weight_p[k]

                    pmf = pmf.cpu().numpy()
                    pmf_clip = np.clip(pmf, 1.0 / 65536, 1.0)
                    pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                    cdf = list(np.add.accumulate(pmf_clip))
                    cdf = [0] + [int(j) for j in cdf]
                    symbol = int(y_hat_np[0, ch, h, w] + minmax)
                    encoder.encode([symbol], cdf)
                    y_hat[0, ch, h + padding, w + padding] = symbol - minmax

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        encoder.close()
        torch.backends.cudnn.deterministic = False

        return {
            "y_stream": y_stream_path,
            "z_stream": z_stream_path,
        }

    def decompress(self, stream_name):
        """
        https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/network.py
        When batch_size is 1
        """
        torch.backends.cudnn.deterministic = True

        # path to save codestreams
        z_stream_path = os.path.join(stream_name + '.npz')
        y_stream_path = os.path.join(stream_name + '.bin')

        fileobj = open(z_stream_path, mode='rb')
        x_shape = np.frombuffer(fileobj.read(4), dtype=np.uint16)
        z_length, minmax = np.frombuffer(fileobj.read(4), dtype=np.uint16)
        flag_nums = np.frombuffer(fileobj.read(self.N // 8), dtype=np.uint8)
        z_strings = fileobj.read(z_length)
        fileobj.close()

        flag = np.unpackbits(flag_nums)
        non_zero_idx = np.squeeze(np.where(flag == 1))
        y_shape = x_shape // 16
        z_shape = y_shape // 4
        H, W = y_shape

        z_hat = self.entropy_bottleneck.decompress([z_strings], z_shape)
        params = self.h_s(z_hat)

        kernel_size = 5
        padding = (kernel_size - 1) // 2
        decoder = RangeDecoder(y_stream_path)
        samples = torch.arange(0, minmax * 2 + 1).float().to(z_hat.device)

        y_hat = torch.zeros(
            (1, self.N, H + 2 * padding, W + 2 * padding),
            device=z_hat.device,
        )

        for h in tqdm(range(H)):
            for w in range(W):
                y_crop = y_hat[0: 1, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    weight=self.context_prediction.weight,
                    bias=self.context_prediction.bias
                )

                p = params[0: 1, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_crop, means_crop, weights_crop = gaussian_params.chunk(3, 1)
                scales_crop = scales_crop.reshape(self.K, self.N)
                means_crop = means_crop.reshape(self.K, self.N) + minmax
                weights_crop = weights_crop.reshape(self.K, self.N)
                weights_crop = F.softmax(weights_crop, dim=0)

                for i in range(len(non_zero_idx)):
                    c = non_zero_idx[i]
                    scale_p = scales_crop[:, c]
                    mean_p = means_crop[:, c]
                    weight_p = weights_crop[:, c]

                    pmf = torch.zeros_like(samples)
                    for k in range(self.K):
                        half = float(0.5)
                        value = abs(samples - mean_p[k])
                        scale = self.gaussian_conditional.lower_bound_scale(scale_p[k])
                        upper = self.gaussian_conditional._standardized_cumulative((half - value) / scale)
                        lower = self.gaussian_conditional._standardized_cumulative((-half - value) / scale)
                        pmf += (upper - lower) * weight_p[k]

                    pmf = pmf.cpu().numpy()
                    pmf_clip = np.clip(pmf, 1.0/65536, 1.0)
                    pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                    cdf = list(np.add.accumulate(pmf_clip))
                    cdf = [0] + [int(j) for j in cdf]
                    symbol = decoder.decode(1, cdf)[0]
                    y_hat[0, c, h + padding, w + padding] = symbol - minmax

        decoder.close()
        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp(0, 1)
        torch.backends.cudnn.deterministic = False

        return {'x_hat': x_hat}


class EfficientJPEGRecompression_PloyLap_Large(CompressionModel):
    """Efficient JPEG recompression model.
    Args:
        N (int): Number of channels
        chunk (tuple): chunk type  such as ("scales", "means")
    """

    def __init__(self, N=192, M=288, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.M = M

        self.frequency = [28, 8, 7, 6, 5, 4, 3, 2, 1]
        cumulative_sum = list(accumulate(self.frequency, initial=0))  # [0, 28, 36, 43, 49, 54, 58, 61, 63, 64]
        in_channels_Y1 = [N + sum_val for sum_val in cumulative_sum[:-1]]  # 去掉最后一个元素
        in_channels_Y234 = [N + 3*sum_val for sum_val in cumulative_sum[:-1]]  # 去掉最后一个元素

        self.gaussian_latent_encode = PolynomialLaplaceConditionalLatentCodec_ST

        h_e_Y = nn.Sequential(
            conv(64*4, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=3),
        )

        h_d_Y = nn.Sequential(
            deconv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            deconv(N, M, stride=2, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            deconv(M, N, stride=2, kernel_size=3),
        )

        h_e_C = nn.Sequential(
            conv(2, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=3),
        )

        h_d_C = nn.Sequential(
            deconv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            deconv(N, M, stride=2, kernel_size=3),
        )

        entropy_parameters_CbCr_anchor = nn.Sequential(
            conv(M, N, kernel_size=1, stride=1),
            conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, 3*4, kernel_size=3, stride=1),
        )

        entropy_parameters_CbCr_non_anchor = nn.Sequential(
            conv(M+4, N, kernel_size=1, stride=1),
            conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, 3*4, kernel_size=3, stride=1),
        )

        self.entropy_aprameters_prior = nn.Sequential(
            conv(N+64, N, kernel_size=1, stride=1),
            conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=1),
        )

        self.hyper_cbcr= HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_e_C,
                    h_s=h_d_C,
                    quantizer="ste",
                )
        self.hyper_Y= HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_e_Y,
                    h_s=h_d_Y,
                    quantizer="ste",
                )
        
        self.Gaussion_Ys = nn.ModuleList([
            self._make_Gaussian_entropy_module(i, 3*f, channel=N) for i,  f in zip(in_channels_Y1, self.frequency)
        ])
        self.Gaussion_Ys_2 = nn.ModuleList([
            self._make_Gaussian_entropy_module(i, 3*f, channel=N) for i,  f in zip(in_channels_Y234, self.frequency)
        ])

        self.Gaussion_Ys_3 = nn.ModuleList([
            self._make_Gaussian_entropy_module(i, 3*f, channel=N) for i,  f in zip(in_channels_Y234, self.frequency)
        ])

        self.Gaussion_Ys_4 = nn.ModuleList([
            self._make_Gaussian_entropy_module(i, 3*f, channel=N) for i,  f in zip(in_channels_Y234, self.frequency)
        ])

        self.Guassian_cbcr_anchor = self.gaussian_latent_encode(entropy_parameters=entropy_parameters_CbCr_anchor)
        self.Guassian_cbcr_non_anchor = self.gaussian_latent_encode(entropy_parameters=entropy_parameters_CbCr_non_anchor)
    
    def _make_Gaussian_entropy_module(self, in_channels, out_channels, channel):
        entropy_aprameters = nn.Sequential(
            conv(in_channels, channel, kernel_size=1, stride=1),
            conv(channel, channel, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(channel, channel, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(channel, channel, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(channel, out_channels, kernel_size=3, stride=1),
        )
        return self.gaussian_latent_encode(entropy_parameters=entropy_aprameters)

    def split_CbCr(self, CbCr):
        b, c, h, w = CbCr.size()
        CbCr = CbCr.reshape(b, c, h // 2, 2, w // 2, 2)
        CbCr = CbCr.permute(0, 1, 2, 4, 3, 5)
        CbCr_anchor = torch.cat((CbCr[:, :, :, :, 0, 1], CbCr[:, :, :, :, 1, 0]), dim=1)
        CbCr_non_anchor = torch.cat((CbCr[:, :, :, :, 0, 0], CbCr[:, :, :, :, 1, 1]), dim=1)
        return CbCr_anchor, CbCr_non_anchor
    
    def merge_CbCr(self, CbCr_anchor, CbCr_non_anchor):
        CbCr = torch.stack(
            [
                torch.stack([CbCr_non_anchor[:, 0:2], CbCr_anchor[:, 0:2]], dim=-1), 
                torch.stack([CbCr_anchor[:, 2:], CbCr_non_anchor[:, 2:]], dim=-1)
            ],
            dim=-2  
        )
        CbCr = CbCr.permute(0, 1, 2, 4, 3, 5)  
        b, c, h_half, w_half = CbCr_anchor.shape
        return CbCr.reshape(b, c//2, h_half * 2, w_half * 2)

    def split_Y(self, Y):
        b, c, h, w = Y.size()
        Y = Y.reshape(b, c, h // 2, 2, w // 2, 2)
        Y = Y.permute(0, 1, 2, 4, 3, 5)
        return Y[:, :, :, :, 0, 0], Y[:, :, :, :, 0, 1], Y[:, :, :, :, 1, 0], Y[:, :, :, :, 1, 1]
    
    def merge_Y(self, Y1, Y2, Y3, Y4):
        Y_combined = torch.stack(
            [
                torch.stack([Y1, Y2], dim=-1), 
                torch.stack([Y3, Y4], dim=-1)
            ],
            dim=-2  
        )
        Y = Y_combined.permute(0, 1, 2, 4, 3, 5)  
        b, c, h_half, w_half = Y1.shape
        return Y.reshape(b, c, h_half * 2, w_half * 2)
    
    def bpp_loss(self, likelihoods): 
        num_pixels = likelihoods.size(0) * likelihoods.size(2) * likelihoods.size(3)
        return torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)

    def forward(self, Y, Cb, Cr):  # Y b x 32 x 32 x 64
        bpp_likelihoods_z = 0
        bpp_likelihoods_y = 0
        bpp_likelihoods_cbcr = 0
        Y = Y.permute(0, 3, 1, 2) # b x 64 x 32 x 32
        Y1, Y2, Y3, Y4 = self.split_Y(Y) # Y1 b x 64 x 16 x 16 
        CbCr = torch.cat((Cb, Cr), dim=1) # b x 2 x 256 x 128
        cbcr_out = self.hyper_cbcr(CbCr)
        z_cbcr_likelihoods = cbcr_out["likelihoods"]["z"]
        
        h_cbcr = cbcr_out["params"] # b x 288 x 128 x 64
        CbCr_anchor, CbCr_non_anchor = self.split_CbCr(CbCr) # b x 4 x 128 x 64
        cbcr_anchor_out = self.Guassian_cbcr_anchor(CbCr_anchor, h_cbcr)
        cbcr_anchor_likelihoods = cbcr_anchor_out["likelihoods"]["y"]
        # cbcr_anchor_hat = cbcr_anchor_out["y_hat"]
        cbcr_non_anchor_out = self.Guassian_cbcr_non_anchor(CbCr_non_anchor, torch.cat((h_cbcr, CbCr_anchor), dim=1))
        cbcr_non_anchor_likelihoods = cbcr_non_anchor_out["likelihoods"]["y"]
        # cbcr_non_anchor_hat = cbcr_non_anchor_out["y_hat"]

        y_out = self.hyper_Y(torch.cat((Y1, Y2, Y3, Y4), dim=1))
        z_y_likelihoods = y_out["likelihoods"]["z"]
        bpp_likelihoods_z += (self.bpp_loss(z_y_likelihoods) + self.bpp_loss(z_cbcr_likelihoods))
        bpp_likelihoods_cbcr += (self.bpp_loss(cbcr_anchor_likelihoods) + self.bpp_loss(cbcr_non_anchor_likelihoods))
        h_y = y_out["params"] # b x 192 x 16 x 16
        Y1_f = Y1.split(self.frequency, dim=1)
        for i, f in enumerate(self.frequency):
            if i == 0:
                y_out = self.Gaussion_Ys[i](Y1_f[i], h_y)
                
            else:
                y_out = self.Gaussion_Ys[i](Y1_f[i], torch.cat((h_y, *Y1_f[:i]), dim=1))
            bpp_likelihoods_y += self.bpp_loss(y_out["likelihoods"]["y"])

        prior_input = torch.cat((h_y, Y1), dim=1)  # b x 192+64 x 16 x 16
        prior_output = self.entropy_aprameters_prior(prior_input) # b x 192 x 16 x 16
        Y2_f = Y2.split(self.frequency, dim=1)
        Y3_f = Y3.split(self.frequency, dim=1)
        Y4_f = Y4.split(self.frequency, dim=1)
        for i, f in enumerate(self.frequency):
            if i == 0:
                y_out2 = self.Gaussion_Ys_2[i](Y2_f[i], prior_output)
                y_out3 = self.Gaussion_Ys_3[i](Y3_f[i], prior_output)
                y_out4 = self.Gaussion_Ys_4[i](Y4_f[i], prior_output)
                
            else:
                y_out2 = self.Gaussion_Ys_2[i](Y2_f[i], torch.cat((prior_output, *Y2_f[:i], *Y3_f[:i], *Y4_f[:i]), dim=1))
                y_out3 = self.Gaussion_Ys_3[i](Y3_f[i], torch.cat((prior_output, *Y2_f[:i], *Y3_f[:i], *Y4_f[:i]), dim=1))
                y_out4 = self.Gaussion_Ys_4[i](Y4_f[i], torch.cat((prior_output, *Y2_f[:i], *Y3_f[:i], *Y4_f[:i]), dim=1))
            bpp_likelihoods_y += self.bpp_loss(y_out2["likelihoods"]["y"])
            bpp_likelihoods_y += self.bpp_loss(y_out3["likelihoods"]["y"])
            bpp_likelihoods_y += self.bpp_loss(y_out4["likelihoods"]["y"])


        return {"bpp_loss": bpp_likelihoods_z + bpp_likelihoods_y + bpp_likelihoods_cbcr}



    def compress(self, Y, Cb, Cr):
        Y = Y.permute(0, 3, 1, 2) # b x 64 x 32 x 32
        Y1, Y2, Y3, Y4 = self.split_Y(Y) # Y1 b x 64 x 16 x 16 
        CbCr = torch.cat((Cb, Cr), dim=1) # b x 2 x 256 x 128
        z_cbcr_out = self.hyper_cbcr.compress(CbCr)  # string, shape, params
        h_cbcr = z_cbcr_out["params"] # b x 288 x 128 x 64
        CbCr_anchor, CbCr_non_anchor = self.split_CbCr(CbCr) # b x 4 x 128 x 64
        cbcr_anchor_out = self.Guassian_cbcr_anchor.compress(CbCr_anchor, h_cbcr) # string, shape, y_hat
        cbcr_non_anchor_out = self.Guassian_cbcr_non_anchor.compress(CbCr_non_anchor, torch.cat((h_cbcr, CbCr_anchor), dim=1))

        z_y_out = self.hyper_Y.compress(torch.cat((Y1, Y2, Y3, Y4), dim=1))# string, shape, params
        h_y = z_y_out["params"] # b x 192 x 16 x 16
        Y1_f = Y1.split(self.frequency, dim=1)
        y1_outs = []
        for i, f in enumerate(self.frequency):
            if i == 0:
                y1_out = self.Gaussion_Ys[i].compress(Y1_f[i], h_y)
                
            else:
                y1_out = self.Gaussion_Ys[i].compress(Y1_f[i], torch.cat((h_y, *Y1_f[:i]), dim=1))
            y1_outs.append(y1_out)

        prior_input = torch.cat((h_y, Y1), dim=1)  # b x 192+64 x 16 x 16
        prior_output = self.entropy_aprameters_prior(prior_input) # b x 192 x 16 x 16
        Y2_f = Y2.split(self.frequency, dim=1)
        Y3_f = Y3.split(self.frequency, dim=1)
        Y4_f = Y4.split(self.frequency, dim=1)
        y234_outs = []
        for i, f in enumerate(self.frequency):
            if i == 0:
                y234_out = self.Gaussion_Ys_234[i].compress(torch.cat((Y2_f[i], Y3_f[i], Y4_f[i]), dim=1), prior_output)
                
            else:
                y234_out = self.Gaussion_Ys_234[i].compress(torch.cat((Y2_f[i], Y3_f[i], Y4_f[i]), dim=1), 
                                                torch.cat((prior_output, *Y2_f[:i], *Y3_f[:i], *Y4_f[:i]), dim=1))
            y234_outs.append(y234_out)
        
        return {
            "z_cbcr": z_cbcr_out,
            "z_y": z_y_out,
            "cbcr_anchor": cbcr_anchor_out,
            "cbcr_non_anchor": cbcr_non_anchor_out,
            "y1": y1_outs,
            "y234": y234_outs,
        }

    def decompress(self, out_enc):
        z_cbcr = out_enc["z_cbcr"]
        z_y = out_enc["z_y"]
        cbcr_anchor_out = out_enc["cbcr_anchor"]
        cbcr_non_anchor_out = out_enc["cbcr_non_anchor"]
        y1_outs = out_enc["y1"]
        y234_outs = out_enc["y234"]
        z_CbCr = self.hyper_cbcr.decompress(z_cbcr['strings'], z_cbcr['shape'])
        cbcr_anchor_out = self.Guassian_cbcr_anchor.decompress(cbcr_anchor_out['strings'], cbcr_anchor_out['shape'], z_CbCr['params']) # string, shape, y_hat
        cbcr_non_anchor_out = self.Guassian_cbcr_non_anchor.decompress(cbcr_non_anchor_out['strings'], cbcr_non_anchor_out['shape'], 
                                                                       torch.cat((z_CbCr['params'], cbcr_anchor_out['y_hat']), dim=1))
        z_y = self.hyper_Y.decompress(z_y['strings'], z_y['shape'])
        y1_hats = []
        for i, f in enumerate(self.frequency):
            if i == 0:
                y1_hat = self.Gaussion_Ys[i].decompress(y1_outs[i]['strings'], y1_outs[i]['shape'], z_y['params'])
                
            else:
                y1_hat = self.Gaussion_Ys[i].decompress(y1_outs[i]['strings'], y1_outs[i]['shape'], torch.cat((z_y['params'], *y1_hats[:i]), dim=1))
            y1_hats.append(y1_hat['y_hat'])

        y1_hats = torch.cat(y1_hats, dim=1)
        prior_input = torch.cat((z_y['params'], y1_hats), dim=1)  # b x 192+64 x 16 x 16
        prior_output = self.entropy_aprameters_prior(prior_input) # b x 192 x 16 x 16
        y2_hats = []
        y3_hats = []
        y4_hats = []
        for i, f in enumerate(self.frequency):
            if i == 0:
                y234_hat = self.Gaussion_Ys_234[i].decompress(y234_outs[i]['strings'], y234_outs[i]['shape'], prior_output)
                y2_hat, y3_hat, y4_hat = y234_hat['y_hat'].chunk(3, dim=1)
                y2_hats.append(y2_hat)
                y3_hats.append(y3_hat)
                y4_hats.append(y4_hat)
                
            else:
                y234_hat = self.Gaussion_Ys_234[i].decompress(y234_outs[i]['strings'], y234_outs[i]['shape'], 
                                                torch.cat((prior_output, *y2_hats[:i], *y3_hats[:i], *y4_hats[:i]), dim=1))
                y2_hat, y3_hat, y4_hat = y234_hat['y_hat'].chunk(3, dim=1)
                y2_hats.append(y2_hat)
                y3_hats.append(y3_hat)
                y4_hats.append(y4_hat)

        y2_hats = torch.cat(y2_hats, dim=1)
        y3_hats = torch.cat(y3_hats, dim=1)
        y4_hats = torch.cat(y4_hats, dim=1)
        Y_hats = self.merge_Y(y1_hats, y2_hats, y3_hats, y4_hats)
        CbCr_hats = self.merge_CbCr(cbcr_anchor_out['y_hat'], cbcr_non_anchor_out['y_hat'])
        Cb_hats, Cr_hats = torch.chunk(CbCr_hats, 2, dim=1)
        return Y_hats.permute(0, 2, 3, 1).int(), Cb_hats.int(), Cr_hats.int()


# class EfficientJPEGRecompression_PloyLap(CompressionModel):
#     """Efficient JPEG recompression model.
#     Args:
#         N (int): Number of channels
#         chunk (tuple): chunk type  such as ("scales", "means")
#     """

#     def __init__(self, N=192, M=288, **kwargs):
#         super().__init__(**kwargs)
#         self.N = N
#         self.M = M

#         self.frequency = [28, 8, 7, 6, 5, 4, 3, 2, 1]
#         cumulative_sum = list(accumulate(self.frequency, initial=0))  # [0, 28, 36, 43, 49, 54, 58, 61, 63, 64]
#         in_channels_Y1 = [N + sum_val for sum_val in cumulative_sum[:-1]]  # 去掉最后一个元素
#         in_channels_Y234 = [N + 3*sum_val for sum_val in cumulative_sum[:-1]]  # 去掉最后一个元素

#         self.gaussian_latent_encode = PolynomialLaplaceConditionalLatentCodec_ST

#         h_e_Y = nn.Sequential(
#             conv(64*4, N, stride=1, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=3),
#         )

#         h_d_Y = nn.Sequential(
#             deconv(N, N, stride=1, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             deconv(N, M, stride=2, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             deconv(M, N, stride=2, kernel_size=3),
#         )

#         h_e_C = nn.Sequential(
#             conv(2, N, stride=1, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=3),
#         )

#         h_d_C = nn.Sequential(
#             deconv(N, N, stride=1, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             deconv(N, M, stride=2, kernel_size=3),
#         )

#         entropy_parameters_CbCr_anchor = nn.Sequential(
#             conv(M, N, kernel_size=1, stride=1),
#             conv(N, N, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(N, N, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(N, 3*4, kernel_size=3, stride=1),
#         )

#         entropy_parameters_CbCr_non_anchor = nn.Sequential(
#             conv(M+4, N, kernel_size=1, stride=1),
#             conv(N, N, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(N, N, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(N, 3*4, kernel_size=3, stride=1),
#         )

#         self.entropy_aprameters_prior = nn.Sequential(
#             conv(N+64, N, kernel_size=1, stride=1),
#             conv(N, N, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(N, N, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(N, N, kernel_size=3, stride=1),
#         )

#         self.hyper_cbcr= HyperLatentCodec(
#                     entropy_bottleneck=EntropyBottleneck(N),
#                     h_a=h_e_C,
#                     h_s=h_d_C,
#                     quantizer="ste",
#                 )
#         self.hyper_Y= HyperLatentCodec(
#                     entropy_bottleneck=EntropyBottleneck(N),
#                     h_a=h_e_Y,
#                     h_s=h_d_Y,
#                     quantizer="ste",
#                 )
        
#         self.Gaussion_Ys = nn.ModuleList([
#             self._make_Gaussian_entropy_module(i, 3*f, channel=N) for i,  f in zip(in_channels_Y1, self.frequency)
#         ])
#         self.Gaussion_Ys_2 = nn.ModuleList([
#             self._make_Gaussian_entropy_module(i, 3*f, channel=N) for i,  f in zip(in_channels_Y234, self.frequency)
#         ])

#         self.Gaussion_Ys_3 = nn.ModuleList([
#             self._make_Gaussian_entropy_module(i, 3*f, channel=N) for i,  f in zip(in_channels_Y234, self.frequency)
#         ])

#         self.Gaussion_Ys_4 = nn.ModuleList([
#             self._make_Gaussian_entropy_module(i, 3*f, channel=N) for i,  f in zip(in_channels_Y234, self.frequency)
#         ])

#         self.Guassian_cbcr_anchor = self.gaussian_latent_encode(entropy_parameters=entropy_parameters_CbCr_anchor)
#         self.Guassian_cbcr_non_anchor = self.gaussian_latent_encode(entropy_parameters=entropy_parameters_CbCr_non_anchor)
    
#     def _make_Gaussian_entropy_module(self, in_channels, out_channels, channel):
#         entropy_aprameters = nn.Sequential(
#             conv(in_channels, channel, kernel_size=1, stride=1),
#             conv(channel, channel, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(channel, channel, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(channel, out_channels, kernel_size=3, stride=1),
#         )
#         return self.gaussian_latent_encode(entropy_parameters=entropy_aprameters)

#     def split_CbCr(self, CbCr):
#         b, c, h, w = CbCr.size()
#         CbCr = CbCr.reshape(b, c, h // 2, 2, w // 2, 2)
#         CbCr = CbCr.permute(0, 1, 2, 4, 3, 5)
#         CbCr_anchor = torch.cat((CbCr[:, :, :, :, 0, 1], CbCr[:, :, :, :, 1, 0]), dim=1)
#         CbCr_non_anchor = torch.cat((CbCr[:, :, :, :, 0, 0], CbCr[:, :, :, :, 1, 1]), dim=1)
#         return CbCr_anchor, CbCr_non_anchor
    
#     def merge_CbCr(self, CbCr_anchor, CbCr_non_anchor):
#         CbCr = torch.stack(
#             [
#                 torch.stack([CbCr_non_anchor[:, 0:2], CbCr_anchor[:, 0:2]], dim=-1), 
#                 torch.stack([CbCr_anchor[:, 2:], CbCr_non_anchor[:, 2:]], dim=-1)
#             ],
#             dim=-2  
#         )
#         CbCr = CbCr.permute(0, 1, 2, 4, 3, 5)  
#         b, c, h_half, w_half = CbCr_anchor.shape
#         return CbCr.reshape(b, c//2, h_half * 2, w_half * 2)

#     def split_Y(self, Y):
#         b, c, h, w = Y.size()
#         Y = Y.reshape(b, c, h // 2, 2, w // 2, 2)
#         Y = Y.permute(0, 1, 2, 4, 3, 5)
#         return Y[:, :, :, :, 0, 0], Y[:, :, :, :, 0, 1], Y[:, :, :, :, 1, 0], Y[:, :, :, :, 1, 1]
    
#     def merge_Y(self, Y1, Y2, Y3, Y4):
#         Y_combined = torch.stack(
#             [
#                 torch.stack([Y1, Y2], dim=-1), 
#                 torch.stack([Y3, Y4], dim=-1)
#             ],
#             dim=-2  
#         )
#         Y = Y_combined.permute(0, 1, 2, 4, 3, 5)  
#         b, c, h_half, w_half = Y1.shape
#         return Y.reshape(b, c, h_half * 2, w_half * 2)
    
#     def bpp_loss(self, likelihoods): 
#         num_pixels = likelihoods.size(0) * likelihoods.size(2) * likelihoods.size(3)
#         return torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)

#     def forward(self, Y, Cb, Cr):  # Y b x 32 x 32 x 64
#         bpp_likelihoods_z = 0
#         bpp_likelihoods_y = 0
#         bpp_likelihoods_cbcr = 0
#         Y = Y.permute(0, 3, 1, 2) # b x 64 x 32 x 32
#         Y1, Y2, Y3, Y4 = self.split_Y(Y) # Y1 b x 64 x 16 x 16 
#         CbCr = torch.cat((Cb, Cr), dim=1) # b x 2 x 256 x 128
#         cbcr_out = self.hyper_cbcr(CbCr)
#         z_cbcr_likelihoods = cbcr_out["likelihoods"]["z"]
        
#         h_cbcr = cbcr_out["params"] # b x 288 x 128 x 64
#         CbCr_anchor, CbCr_non_anchor = self.split_CbCr(CbCr) # b x 4 x 128 x 64
#         cbcr_anchor_out = self.Guassian_cbcr_anchor(CbCr_anchor, h_cbcr)
#         cbcr_anchor_likelihoods = cbcr_anchor_out["likelihoods"]["y"]
#         # cbcr_anchor_hat = cbcr_anchor_out["y_hat"]
#         cbcr_non_anchor_out = self.Guassian_cbcr_non_anchor(CbCr_non_anchor, torch.cat((h_cbcr, CbCr_anchor), dim=1))
#         cbcr_non_anchor_likelihoods = cbcr_non_anchor_out["likelihoods"]["y"]
#         # cbcr_non_anchor_hat = cbcr_non_anchor_out["y_hat"]

#         y_out = self.hyper_Y(torch.cat((Y1, Y2, Y3, Y4), dim=1))
#         z_y_likelihoods = y_out["likelihoods"]["z"]
#         bpp_likelihoods_z += (self.bpp_loss(z_y_likelihoods) + self.bpp_loss(z_cbcr_likelihoods))
#         bpp_likelihoods_cbcr += (self.bpp_loss(cbcr_anchor_likelihoods) + self.bpp_loss(cbcr_non_anchor_likelihoods))
#         h_y = y_out["params"] # b x 192 x 16 x 16
#         Y1_f = Y1.split(self.frequency, dim=1)
#         for i, f in enumerate(self.frequency):
#             if i == 0:
#                 y_out = self.Gaussion_Ys[i](Y1_f[i], h_y)
                
#             else:
#                 y_out = self.Gaussion_Ys[i](Y1_f[i], torch.cat((h_y, *Y1_f[:i]), dim=1))
#             bpp_likelihoods_y += self.bpp_loss(y_out["likelihoods"]["y"])

#         prior_input = torch.cat((h_y, Y1), dim=1)  # b x 192+64 x 16 x 16
#         prior_output = self.entropy_aprameters_prior(prior_input) # b x 192 x 16 x 16
#         Y2_f = Y2.split(self.frequency, dim=1)
#         Y3_f = Y3.split(self.frequency, dim=1)
#         Y4_f = Y4.split(self.frequency, dim=1)
#         for i, f in enumerate(self.frequency):
#             if i == 0:
#                 y_out2 = self.Gaussion_Ys_2[i](Y2_f[i], prior_output)
#                 y_out3 = self.Gaussion_Ys_3[i](Y3_f[i], prior_output)
#                 y_out4 = self.Gaussion_Ys_4[i](Y4_f[i], prior_output)
                
#             else:
#                 y_out2 = self.Gaussion_Ys_2[i](Y2_f[i], torch.cat((prior_output, *Y2_f[:i], *Y3_f[:i], *Y4_f[:i]), dim=1))
#                 y_out3 = self.Gaussion_Ys_3[i](Y3_f[i], torch.cat((prior_output, *Y2_f[:i], *Y3_f[:i], *Y4_f[:i]), dim=1))
#                 y_out4 = self.Gaussion_Ys_4[i](Y4_f[i], torch.cat((prior_output, *Y2_f[:i], *Y3_f[:i], *Y4_f[:i]), dim=1))
#             bpp_likelihoods_y += self.bpp_loss(y_out2["likelihoods"]["y"])
#             bpp_likelihoods_y += self.bpp_loss(y_out3["likelihoods"]["y"])
#             bpp_likelihoods_y += self.bpp_loss(y_out4["likelihoods"]["y"])


#         return {"bpp_loss": bpp_likelihoods_z + bpp_likelihoods_y + bpp_likelihoods_cbcr}



#     def compress(self, Y, Cb, Cr):
#         Y = Y.permute(0, 3, 1, 2) # b x 64 x 32 x 32
#         Y1, Y2, Y3, Y4 = self.split_Y(Y) # Y1 b x 64 x 16 x 16 
#         CbCr = torch.cat((Cb, Cr), dim=1) # b x 2 x 256 x 128
#         z_cbcr_out = self.hyper_cbcr.compress(CbCr)  # string, shape, params
#         h_cbcr = z_cbcr_out["params"] # b x 288 x 128 x 64
#         CbCr_anchor, CbCr_non_anchor = self.split_CbCr(CbCr) # b x 4 x 128 x 64
#         cbcr_anchor_out = self.Guassian_cbcr_anchor.compress(CbCr_anchor, h_cbcr) # string, shape, y_hat
#         cbcr_non_anchor_out = self.Guassian_cbcr_non_anchor.compress(CbCr_non_anchor, torch.cat((h_cbcr, CbCr_anchor), dim=1))

#         z_y_out = self.hyper_Y.compress(torch.cat((Y1, Y2, Y3, Y4), dim=1))# string, shape, params
#         h_y = z_y_out["params"] # b x 192 x 16 x 16
#         Y1_f = Y1.split(self.frequency, dim=1)
#         y1_outs = []
#         for i, f in enumerate(self.frequency):
#             if i == 0:
#                 y1_out = self.Gaussion_Ys[i].compress(Y1_f[i], h_y)
                
#             else:
#                 y1_out = self.Gaussion_Ys[i].compress(Y1_f[i], torch.cat((h_y, *Y1_f[:i]), dim=1))
#             y1_outs.append(y1_out)

#         prior_input = torch.cat((h_y, Y1), dim=1)  # b x 192+64 x 16 x 16
#         prior_output = self.entropy_aprameters_prior(prior_input) # b x 192 x 16 x 16
#         Y2_f = Y2.split(self.frequency, dim=1)
#         Y3_f = Y3.split(self.frequency, dim=1)
#         Y4_f = Y4.split(self.frequency, dim=1)
#         y234_outs = []
#         for i, f in enumerate(self.frequency):
#             if i == 0:
#                 y234_out = self.Gaussion_Ys_234[i].compress(torch.cat((Y2_f[i], Y3_f[i], Y4_f[i]), dim=1), prior_output)
                
#             else:
#                 y234_out = self.Gaussion_Ys_234[i].compress(torch.cat((Y2_f[i], Y3_f[i], Y4_f[i]), dim=1), 
#                                                 torch.cat((prior_output, *Y2_f[:i], *Y3_f[:i], *Y4_f[:i]), dim=1))
#             y234_outs.append(y234_out)
        
#         return {
#             "z_cbcr": z_cbcr_out,
#             "z_y": z_y_out,
#             "cbcr_anchor": cbcr_anchor_out,
#             "cbcr_non_anchor": cbcr_non_anchor_out,
#             "y1": y1_outs,
#             "y234": y234_outs,
#         }

#     def decompress(self, out_enc):
#         z_cbcr = out_enc["z_cbcr"]
#         z_y = out_enc["z_y"]
#         cbcr_anchor_out = out_enc["cbcr_anchor"]
#         cbcr_non_anchor_out = out_enc["cbcr_non_anchor"]
#         y1_outs = out_enc["y1"]
#         y234_outs = out_enc["y234"]
#         z_CbCr = self.hyper_cbcr.decompress(z_cbcr['strings'], z_cbcr['shape'])
#         cbcr_anchor_out = self.Guassian_cbcr_anchor.decompress(cbcr_anchor_out['strings'], cbcr_anchor_out['shape'], z_CbCr['params']) # string, shape, y_hat
#         cbcr_non_anchor_out = self.Guassian_cbcr_non_anchor.decompress(cbcr_non_anchor_out['strings'], cbcr_non_anchor_out['shape'], 
#                                                                        torch.cat((z_CbCr['params'], cbcr_anchor_out['y_hat']), dim=1))
#         z_y = self.hyper_Y.decompress(z_y['strings'], z_y['shape'])
#         y1_hats = []
#         for i, f in enumerate(self.frequency):
#             if i == 0:
#                 y1_hat = self.Gaussion_Ys[i].decompress(y1_outs[i]['strings'], y1_outs[i]['shape'], z_y['params'])
                
#             else:
#                 y1_hat = self.Gaussion_Ys[i].decompress(y1_outs[i]['strings'], y1_outs[i]['shape'], torch.cat((z_y['params'], *y1_hats[:i]), dim=1))
#             y1_hats.append(y1_hat['y_hat'])

#         y1_hats = torch.cat(y1_hats, dim=1)
#         prior_input = torch.cat((z_y['params'], y1_hats), dim=1)  # b x 192+64 x 16 x 16
#         prior_output = self.entropy_aprameters_prior(prior_input) # b x 192 x 16 x 16
#         y2_hats = []
#         y3_hats = []
#         y4_hats = []
#         for i, f in enumerate(self.frequency):
#             if i == 0:
#                 y234_hat = self.Gaussion_Ys_234[i].decompress(y234_outs[i]['strings'], y234_outs[i]['shape'], prior_output)
#                 y2_hat, y3_hat, y4_hat = y234_hat['y_hat'].chunk(3, dim=1)
#                 y2_hats.append(y2_hat)
#                 y3_hats.append(y3_hat)
#                 y4_hats.append(y4_hat)
                
#             else:
#                 y234_hat = self.Gaussion_Ys_234[i].decompress(y234_outs[i]['strings'], y234_outs[i]['shape'], 
#                                                 torch.cat((prior_output, *y2_hats[:i], *y3_hats[:i], *y4_hats[:i]), dim=1))
#                 y2_hat, y3_hat, y4_hat = y234_hat['y_hat'].chunk(3, dim=1)
#                 y2_hats.append(y2_hat)
#                 y3_hats.append(y3_hat)
#                 y4_hats.append(y4_hat)

#         y2_hats = torch.cat(y2_hats, dim=1)
#         y3_hats = torch.cat(y3_hats, dim=1)
#         y4_hats = torch.cat(y4_hats, dim=1)
#         Y_hats = self.merge_Y(y1_hats, y2_hats, y3_hats, y4_hats)
#         CbCr_hats = self.merge_CbCr(cbcr_anchor_out['y_hat'], cbcr_non_anchor_out['y_hat'])
#         Cb_hats, Cr_hats = torch.chunk(CbCr_hats, 2, dim=1)
#         return Y_hats.permute(0, 2, 3, 1).int(), Cb_hats.int(), Cr_hats.int()


    
# class EfficientJPEGRecompression_PloyLap(CompressionModel):
#     """Efficient JPEG recompression model.
#     Args:
#         N (int): Number of channels
#         chunk (tuple): chunk type  such as ("scales", "means")
#     """

#     def __init__(self, N=192, M=288, **kwargs):
#         super().__init__(**kwargs)
#         self.N = N
#         self.M = M

#         self.frequency = [28, 8, 7, 6, 5, 4, 3, 2, 1]
#         cumulative_sum = list(accumulate(self.frequency, initial=0))  # [0, 28, 36, 43, 49, 54, 58, 61, 63, 64]
#         in_channels_Y1 = [N + sum_val for sum_val in cumulative_sum[:-1]]  # 去掉最后一个元素
#         in_channels_Y234 = [N + 3*sum_val for sum_val in cumulative_sum[:-1]]  # 去掉最后一个元素

#         self.gaussian_latent_encode = PolynomialLaplaceConditionalLatentCodec_ST

#         h_e_Y = nn.Sequential(
#             conv(64*4, N, stride=1, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=3),
#         )

#         h_d_Y = nn.Sequential(
#             deconv(N, N, stride=1, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             deconv(N, M, stride=2, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             deconv(M, N, stride=2, kernel_size=3),
#         )

#         h_e_C = nn.Sequential(
#             conv(2, N, stride=1, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=3),
#         )

#         h_d_C = nn.Sequential(
#             deconv(N, N, stride=1, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             deconv(N, M, stride=2, kernel_size=3),
#         )

#         entropy_parameters_CbCr_anchor = nn.Sequential(
#             conv(M, N, kernel_size=1, stride=1),
#             conv(N, N, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(N, N, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(N, 3*4, kernel_size=3, stride=1),
#         )

#         entropy_parameters_CbCr_non_anchor = nn.Sequential(
#             conv(M+4, N, kernel_size=1, stride=1),
#             conv(N, N, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(N, N, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(N, 3*4, kernel_size=3, stride=1),
#         )

#         self.entropy_aprameters_prior = nn.Sequential(
#             conv(N+64, N, kernel_size=1, stride=1),
#             conv(N, N, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(N, N, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(N, N, kernel_size=3, stride=1),
#         )

#         self.hyper_cbcr= HyperLatentCodec(
#                     entropy_bottleneck=EntropyBottleneck(N),
#                     h_a=h_e_C,
#                     h_s=h_d_C,
#                     quantizer="ste",
#                 )
#         self.hyper_Y= HyperLatentCodec(
#                     entropy_bottleneck=EntropyBottleneck(N),
#                     h_a=h_e_Y,
#                     h_s=h_d_Y,
#                     quantizer="ste",
#                 )
        
#         self.Gaussion_Ys = nn.ModuleList([
#             self._make_Gaussian_entropy_module(i, f, channel=N) for i,  f in zip(in_channels_Y1, self.frequency)
#         ])
#         self.Gaussion_Ys_234 = nn.ModuleList([
#             self._make_Gaussian_entropy_module(i, 3*f, channel=N) for i,  f in zip(in_channels_Y234, self.frequency)
#         ])


#         self.Guassian_cbcr_anchor = self.gaussian_latent_encode(entropy_parameters=entropy_parameters_CbCr_anchor)
#         self.Guassian_cbcr_non_anchor = self.gaussian_latent_encode(entropy_parameters=entropy_parameters_CbCr_non_anchor)
    
#     def _make_Gaussian_entropy_module(self, in_channels, out_channels, channel):
#         entropy_aprameters = nn.Sequential(
#             conv(in_channels, channel, kernel_size=1, stride=1),
#             conv(channel, channel, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(channel, channel, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True),
#             conv(channel, 3*out_channels, kernel_size=3, stride=1),
#         )
#         return self.gaussian_latent_encode(entropy_parameters=entropy_aprameters)

#     def split_CbCr(self, CbCr):
#         b, c, h, w = CbCr.size()
#         CbCr = CbCr.reshape(b, c, h // 2, 2, w // 2, 2)
#         CbCr = CbCr.permute(0, 1, 2, 4, 3, 5)
#         CbCr_anchor = torch.cat((CbCr[:, :, :, :, 0, 1], CbCr[:, :, :, :, 1, 0]), dim=1)
#         CbCr_non_anchor = torch.cat((CbCr[:, :, :, :, 0, 0], CbCr[:, :, :, :, 1, 1]), dim=1)
#         return CbCr_anchor, CbCr_non_anchor
    
#     def merge_CbCr(self, CbCr_anchor, CbCr_non_anchor):
#         CbCr = torch.stack(
#             [
#                 torch.stack([CbCr_non_anchor[:, 0:2], CbCr_anchor[:, 0:2]], dim=-1), 
#                 torch.stack([CbCr_anchor[:, 2:], CbCr_non_anchor[:, 2:]], dim=-1)
#             ],
#             dim=-2  
#         )
#         CbCr = CbCr.permute(0, 1, 2, 4, 3, 5)  
#         b, c, h_half, w_half = CbCr_anchor.shape
#         return CbCr.reshape(b, c//2, h_half * 2, w_half * 2)

#     def split_Y(self, Y):
#         b, c, h, w = Y.size()
#         Y = Y.reshape(b, c, h // 2, 2, w // 2, 2)
#         Y = Y.permute(0, 1, 2, 4, 3, 5)
#         return Y[:, :, :, :, 0, 0], Y[:, :, :, :, 0, 1], Y[:, :, :, :, 1, 0], Y[:, :, :, :, 1, 1]
    
#     def merge_Y(self, Y1, Y2, Y3, Y4):
#         Y_combined = torch.stack(
#             [
#                 torch.stack([Y1, Y2], dim=-1), 
#                 torch.stack([Y3, Y4], dim=-1)
#             ],
#             dim=-2  
#         )
#         Y = Y_combined.permute(0, 1, 2, 4, 3, 5)  
#         b, c, h_half, w_half = Y1.shape
#         return Y.reshape(b, c, h_half * 2, w_half * 2)
    
#     def bpp_loss(self, likelihoods): 
#         num_pixels = likelihoods.size(0) * likelihoods.size(2) * likelihoods.size(3)
#         return torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)

#     def forward(self, Y, Cb, Cr):  # Y b x 32 x 32 x 64
#         bpp_likelihoods_z = 0
#         bpp_likelihoods_y = 0
#         bpp_likelihoods_cbcr = 0
#         Y = Y.permute(0, 3, 1, 2) # b x 64 x 32 x 32
#         Y1, Y2, Y3, Y4 = self.split_Y(Y) # Y1 b x 64 x 16 x 16 
#         CbCr = torch.cat((Cb, Cr), dim=1) # b x 2 x 256 x 128
#         cbcr_out = self.hyper_cbcr(CbCr)
#         z_cbcr_likelihoods = cbcr_out["likelihoods"]["z"]
        
#         h_cbcr = cbcr_out["params"] # b x 288 x 128 x 64
#         CbCr_anchor, CbCr_non_anchor = self.split_CbCr(CbCr) # b x 4 x 128 x 64
#         cbcr_anchor_out = self.Guassian_cbcr_anchor(CbCr_anchor, h_cbcr)
#         cbcr_anchor_likelihoods = cbcr_anchor_out["likelihoods"]["y"]
#         # cbcr_anchor_hat = cbcr_anchor_out["y_hat"]
#         cbcr_non_anchor_out = self.Guassian_cbcr_non_anchor(CbCr_non_anchor, torch.cat((h_cbcr, CbCr_anchor), dim=1))
#         cbcr_non_anchor_likelihoods = cbcr_non_anchor_out["likelihoods"]["y"]
#         # cbcr_non_anchor_hat = cbcr_non_anchor_out["y_hat"]

#         y_out = self.hyper_Y(torch.cat((Y1, Y2, Y3, Y4), dim=1))
#         z_y_likelihoods = y_out["likelihoods"]["z"]
#         bpp_likelihoods_z += (self.bpp_loss(z_y_likelihoods) + self.bpp_loss(z_cbcr_likelihoods))
#         bpp_likelihoods_cbcr += (self.bpp_loss(cbcr_anchor_likelihoods) + self.bpp_loss(cbcr_non_anchor_likelihoods))
#         h_y = y_out["params"] # b x 192 x 16 x 16
#         Y1_f = Y1.split(self.frequency, dim=1)
#         for i, f in enumerate(self.frequency):
#             if i == 0:
#                 y_out = self.Gaussion_Ys[i](Y1_f[i], h_y)
                
#             else:
#                 y_out = self.Gaussion_Ys[i](Y1_f[i], torch.cat((h_y, *Y1_f[:i]), dim=1))
#             bpp_likelihoods_y += self.bpp_loss(y_out["likelihoods"]["y"])

#         prior_input = torch.cat((h_y, Y1), dim=1)  # b x 192+64 x 16 x 16
#         prior_output = self.entropy_aprameters_prior(prior_input) # b x 192 x 16 x 16
#         Y2_f = Y2.split(self.frequency, dim=1)
#         Y3_f = Y3.split(self.frequency, dim=1)
#         Y4_f = Y4.split(self.frequency, dim=1)
#         for i, f in enumerate(self.frequency):
#             if i == 0:
#                 y_out = self.Gaussion_Ys_234[i](torch.cat((Y2_f[i], Y3_f[i], Y4_f[i]), dim=1), prior_output)
                
#             else:
#                 y_out = self.Gaussion_Ys_234[i](torch.cat((Y2_f[i], Y3_f[i], Y4_f[i]), dim=1), 
#                                                 torch.cat((prior_output, *Y2_f[:i], *Y3_f[:i], *Y4_f[:i]), dim=1))
#             bpp_likelihoods_y += self.bpp_loss(y_out["likelihoods"]["y"])


#         return {"bpp_loss": bpp_likelihoods_z + bpp_likelihoods_y + bpp_likelihoods_cbcr}



#     def compress(self, Y, Cb, Cr):
#         Y = Y.permute(0, 3, 1, 2) # b x 64 x 32 x 32
#         Y1, Y2, Y3, Y4 = self.split_Y(Y) # Y1 b x 64 x 16 x 16 
#         CbCr = torch.cat((Cb, Cr), dim=1) # b x 2 x 256 x 128
#         z_cbcr_out = self.hyper_cbcr.compress(CbCr)  # string, shape, params
#         h_cbcr = z_cbcr_out["params"] # b x 288 x 128 x 64
#         CbCr_anchor, CbCr_non_anchor = self.split_CbCr(CbCr) # b x 4 x 128 x 64
#         cbcr_anchor_out = self.Guassian_cbcr_anchor.compress(CbCr_anchor, h_cbcr) # string, shape, y_hat
#         cbcr_non_anchor_out = self.Guassian_cbcr_non_anchor.compress(CbCr_non_anchor, torch.cat((h_cbcr, CbCr_anchor), dim=1))

#         z_y_out = self.hyper_Y.compress(torch.cat((Y1, Y2, Y3, Y4), dim=1))# string, shape, params
#         h_y = z_y_out["params"] # b x 192 x 16 x 16
#         Y1_f = Y1.split(self.frequency, dim=1)
#         y1_outs = []
#         for i, f in enumerate(self.frequency):
#             if i == 0:
#                 y1_out = self.Gaussion_Ys[i].compress(Y1_f[i], h_y)
                
#             else:
#                 y1_out = self.Gaussion_Ys[i].compress(Y1_f[i], torch.cat((h_y, *Y1_f[:i]), dim=1))
#             y1_outs.append(y1_out)

#         prior_input = torch.cat((h_y, Y1), dim=1)  # b x 192+64 x 16 x 16
#         prior_output = self.entropy_aprameters_prior(prior_input) # b x 192 x 16 x 16
#         Y2_f = Y2.split(self.frequency, dim=1)
#         Y3_f = Y3.split(self.frequency, dim=1)
#         Y4_f = Y4.split(self.frequency, dim=1)
#         y234_outs = []
#         for i, f in enumerate(self.frequency):
#             if i == 0:
#                 y234_out = self.Gaussion_Ys_234[i].compress(torch.cat((Y2_f[i], Y3_f[i], Y4_f[i]), dim=1), prior_output)
                
#             else:
#                 y234_out = self.Gaussion_Ys_234[i].compress(torch.cat((Y2_f[i], Y3_f[i], Y4_f[i]), dim=1), 
#                                                 torch.cat((prior_output, *Y2_f[:i], *Y3_f[:i], *Y4_f[:i]), dim=1))
#             y234_outs.append(y234_out)
        
#         return {
#             "z_cbcr": z_cbcr_out,
#             "z_y": z_y_out,
#             "cbcr_anchor": cbcr_anchor_out,
#             "cbcr_non_anchor": cbcr_non_anchor_out,
#             "y1": y1_outs,
#             "y234": y234_outs,
#         }

#     def decompress(self, out_enc):
#         z_cbcr = out_enc["z_cbcr"]
#         z_y = out_enc["z_y"]
#         cbcr_anchor_out = out_enc["cbcr_anchor"]
#         cbcr_non_anchor_out = out_enc["cbcr_non_anchor"]
#         y1_outs = out_enc["y1"]
#         y234_outs = out_enc["y234"]
#         z_CbCr = self.hyper_cbcr.decompress(z_cbcr['strings'], z_cbcr['shape'])
#         cbcr_anchor_out = self.Guassian_cbcr_anchor.decompress(cbcr_anchor_out['strings'], cbcr_anchor_out['shape'], z_CbCr['params']) # string, shape, y_hat
#         cbcr_non_anchor_out = self.Guassian_cbcr_non_anchor.decompress(cbcr_non_anchor_out['strings'], cbcr_non_anchor_out['shape'], 
#                                                                        torch.cat((z_CbCr['params'], cbcr_anchor_out['y_hat']), dim=1))
#         z_y = self.hyper_Y.decompress(z_y['strings'], z_y['shape'])
#         y1_hats = []
#         for i, f in enumerate(self.frequency):
#             if i == 0:
#                 y1_hat = self.Gaussion_Ys[i].decompress(y1_outs[i]['strings'], y1_outs[i]['shape'], z_y['params'])
                
#             else:
#                 y1_hat = self.Gaussion_Ys[i].decompress(y1_outs[i]['strings'], y1_outs[i]['shape'], torch.cat((z_y['params'], *y1_hats[:i]), dim=1))
#             y1_hats.append(y1_hat['y_hat'])

#         y1_hats = torch.cat(y1_hats, dim=1)
#         prior_input = torch.cat((z_y['params'], y1_hats), dim=1)  # b x 192+64 x 16 x 16
#         prior_output = self.entropy_aprameters_prior(prior_input) # b x 192 x 16 x 16
#         y2_hats = []
#         y3_hats = []
#         y4_hats = []
#         for i, f in enumerate(self.frequency):
#             if i == 0:
#                 y234_hat = self.Gaussion_Ys_234[i].decompress(y234_outs[i]['strings'], y234_outs[i]['shape'], prior_output)
#                 y2_hat, y3_hat, y4_hat = y234_hat['y_hat'].chunk(3, dim=1)
#                 y2_hats.append(y2_hat)
#                 y3_hats.append(y3_hat)
#                 y4_hats.append(y4_hat)
                
#             else:
#                 y234_hat = self.Gaussion_Ys_234[i].decompress(y234_outs[i]['strings'], y234_outs[i]['shape'], 
#                                                 torch.cat((prior_output, *y2_hats[:i], *y3_hats[:i], *y4_hats[:i]), dim=1))
#                 y2_hat, y3_hat, y4_hat = y234_hat['y_hat'].chunk(3, dim=1)
#                 y2_hats.append(y2_hat)
#                 y3_hats.append(y3_hat)
#                 y4_hats.append(y4_hat)

#         y2_hats = torch.cat(y2_hats, dim=1)
#         y3_hats = torch.cat(y3_hats, dim=1)
#         y4_hats = torch.cat(y4_hats, dim=1)
#         Y_hats = self.merge_Y(y1_hats, y2_hats, y3_hats, y4_hats)
#         CbCr_hats = self.merge_CbCr(cbcr_anchor_out['y_hat'], cbcr_non_anchor_out['y_hat'])
#         Cb_hats, Cr_hats = torch.chunk(CbCr_hats, 2, dim=1)
#         return Y_hats.permute(0, 2, 3, 1).int(), Cb_hats.int(), Cr_hats.int()

@register_model("cheng2020-anchor-checkerboard")
class EfficientJPEGRecompression(CompressionModel):
    """Efficient JPEG recompression model.
    Args:
        N (int): Number of channels
        chunk (tuple): chunk type  such as ("scales", "means")
    """

    def __init__(self, N=192, M=288, chunk= ("scales",), **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.M = M
        self.chunk = chunk

        self.frequency = [28, 8, 7, 6, 5, 4, 3, 2, 1]
        cumulative_sum = list(accumulate(self.frequency, initial=0))  # [0, 28, 36, 43, 49, 54, 58, 61, 63, 64]
        in_channels_Y1 = [N + sum_val for sum_val in cumulative_sum[:-1]]  # 去掉最后一个元素
        in_channels_Y234 = [N + 3*sum_val for sum_val in cumulative_sum[:-1]]  # 去掉最后一个元素

        self.gaussian_latent_encode = GaussianConditionalLatentCodec if chunk == ("scales",) else GaussianConditionalLatentCodec_ST

        h_e_Y = nn.Sequential(
            conv(64*4, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=3),
        )

        h_d_Y = nn.Sequential(
            deconv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            deconv(N, M, stride=2, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            deconv(M, N, stride=2, kernel_size=3),
        )

        h_e_C = nn.Sequential(
            conv(2, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=3),
        )

        h_d_C = nn.Sequential(
            deconv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            deconv(N, M, stride=2, kernel_size=3),
        )

        entropy_parameters_CbCr_anchor = nn.Sequential(
            conv(M, N, kernel_size=1, stride=1),
            conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, len(self.chunk)*4, kernel_size=3, stride=1),
        )

        entropy_parameters_CbCr_non_anchor = nn.Sequential(
            conv(M+4, N, kernel_size=1, stride=1),
            conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, len(self.chunk)*4, kernel_size=3, stride=1),
        )

        self.entropy_aprameters_prior = nn.Sequential(
            conv(N+64, N, kernel_size=1, stride=1),
            conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=1),
        )

        self.hyper_cbcr= HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_e_C,
                    h_s=h_d_C,
                    quantizer="ste",
                )
        self.hyper_Y= HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_e_Y,
                    h_s=h_d_Y,
                    quantizer="ste",
                )
        
        self.Gaussion_Ys = nn.ModuleList([
            self._make_Gaussian_entropy_module(i, f, channel=N) for i,  f in zip(in_channels_Y1, self.frequency)
        ])
        self.Gaussion_Ys_234 = nn.ModuleList([
            self._make_Gaussian_entropy_module(i, 3*f, channel=N) for i,  f in zip(in_channels_Y234, self.frequency)
        ])


        self.Guassian_cbcr_anchor = self.gaussian_latent_encode(chunks=self.chunk,
                                                                   entropy_parameters=entropy_parameters_CbCr_anchor)
        self.Guassian_cbcr_non_anchor = self.gaussian_latent_encode(chunks=self.chunk,
                                                                       entropy_parameters=entropy_parameters_CbCr_non_anchor)
    
    def _make_Gaussian_entropy_module(self, in_channels, out_channels, channel):
        entropy_aprameters = nn.Sequential(
            conv(in_channels, channel, kernel_size=1, stride=1),
            conv(channel, channel, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(channel, channel, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(channel, len(self.chunk)*out_channels, kernel_size=3, stride=1),
        )
        return self.gaussian_latent_encode(chunks=self.chunk, 
                                              entropy_parameters=entropy_aprameters)

    def split_CbCr(self, CbCr):
        b, c, h, w = CbCr.size()
        CbCr = CbCr.reshape(b, c, h // 2, 2, w // 2, 2)
        CbCr = CbCr.permute(0, 1, 2, 4, 3, 5)
        CbCr_anchor = torch.cat((CbCr[:, :, :, :, 0, 1], CbCr[:, :, :, :, 1, 0]), dim=1)
        CbCr_non_anchor = torch.cat((CbCr[:, :, :, :, 0, 0], CbCr[:, :, :, :, 1, 1]), dim=1)
        return CbCr_anchor, CbCr_non_anchor
    
    def merge_CbCr(self, CbCr_anchor, CbCr_non_anchor):
        CbCr = torch.stack(
            [
                torch.stack([CbCr_non_anchor[:, 0:2], CbCr_anchor[:, 0:2]], dim=-1), 
                torch.stack([CbCr_anchor[:, 2:], CbCr_non_anchor[:, 2:]], dim=-1)
            ],
            dim=-2  
        )
        CbCr = CbCr.permute(0, 1, 2, 4, 3, 5)  
        b, c, h_half, w_half = CbCr_anchor.shape
        return CbCr.reshape(b, c//2, h_half * 2, w_half * 2)

    def split_Y(self, Y):
        b, c, h, w = Y.size()
        Y = Y.reshape(b, c, h // 2, 2, w // 2, 2)
        Y = Y.permute(0, 1, 2, 4, 3, 5)
        return Y[:, :, :, :, 0, 0], Y[:, :, :, :, 0, 1], Y[:, :, :, :, 1, 0], Y[:, :, :, :, 1, 1]
    
    def merge_Y(self, Y1, Y2, Y3, Y4):
        Y_combined = torch.stack(
            [
                torch.stack([Y1, Y2], dim=-1), 
                torch.stack([Y3, Y4], dim=-1)
            ],
            dim=-2  
        )
        Y = Y_combined.permute(0, 1, 2, 4, 3, 5)  
        b, c, h_half, w_half = Y1.shape
        return Y.reshape(b, c, h_half * 2, w_half * 2)
    
    def bpp_loss(self, likelihoods):
        num_pixels = likelihoods.size(0) * likelihoods.size(2) * likelihoods.size(3)
        return torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)

    def forward(self, Y, Cb, Cr):  # Y b x 32 x 32 x 64
        bpp_likelihoods_z = 0
        bpp_likelihoods_y = 0
        bpp_likelihoods_cbcr = 0
        Y = Y.permute(0, 3, 1, 2) # b x 64 x 32 x 32
        Y1, Y2, Y3, Y4 = self.split_Y(Y) # Y1 b x 64 x 16 x 16 
        CbCr = torch.cat((Cb, Cr), dim=1) # b x 2 x 256 x 128
        cbcr_out = self.hyper_cbcr(CbCr)
        z_cbcr_likelihoods = cbcr_out["likelihoods"]["z"]
        
        h_cbcr = cbcr_out["params"] # b x 288 x 128 x 64
        CbCr_anchor, CbCr_non_anchor = self.split_CbCr(CbCr) # b x 4 x 128 x 64
        cbcr_anchor_out = self.Guassian_cbcr_anchor(CbCr_anchor, h_cbcr)
        cbcr_anchor_likelihoods = cbcr_anchor_out["likelihoods"]["y"]
        # cbcr_anchor_hat = cbcr_anchor_out["y_hat"]
        cbcr_non_anchor_out = self.Guassian_cbcr_non_anchor(CbCr_non_anchor, torch.cat((h_cbcr, CbCr_anchor), dim=1))
        cbcr_non_anchor_likelihoods = cbcr_non_anchor_out["likelihoods"]["y"]
        # cbcr_non_anchor_hat = cbcr_non_anchor_out["y_hat"]

        y_out = self.hyper_Y(torch.cat((Y1, Y2, Y3, Y4), dim=1))
        z_y_likelihoods = y_out["likelihoods"]["z"]
        bpp_likelihoods_z += (self.bpp_loss(z_y_likelihoods) + self.bpp_loss(z_cbcr_likelihoods))
        bpp_likelihoods_cbcr += (self.bpp_loss(cbcr_anchor_likelihoods) + self.bpp_loss(cbcr_non_anchor_likelihoods))
        h_y = y_out["params"] # b x 192 x 16 x 16
        Y1_f = Y1.split(self.frequency, dim=1)
        for i, f in enumerate(self.frequency):
            if i == 0:
                y_out = self.Gaussion_Ys[i](Y1_f[i], h_y)
                
            else:
                y_out = self.Gaussion_Ys[i](Y1_f[i], torch.cat((h_y, *Y1_f[:i]), dim=1))
            bpp_likelihoods_y += self.bpp_loss(y_out["likelihoods"]["y"])

        prior_input = torch.cat((h_y, Y1), dim=1)  # b x 192+64 x 16 x 16
        prior_output = self.entropy_aprameters_prior(prior_input) # b x 192 x 16 x 16
        Y2_f = Y2.split(self.frequency, dim=1)
        Y3_f = Y3.split(self.frequency, dim=1)
        Y4_f = Y4.split(self.frequency, dim=1)
        for i, f in enumerate(self.frequency):
            if i == 0:
                y_out = self.Gaussion_Ys_234[i](torch.cat((Y2_f[i], Y3_f[i], Y4_f[i]), dim=1), prior_output)
                
            else:
                y_out = self.Gaussion_Ys_234[i](torch.cat((Y2_f[i], Y3_f[i], Y4_f[i]), dim=1), 
                                                torch.cat((prior_output, *Y2_f[:i], *Y3_f[:i], *Y4_f[:i]), dim=1))
            bpp_likelihoods_y += self.bpp_loss(y_out["likelihoods"]["y"])


        return {"bpp_loss": bpp_likelihoods_z + bpp_likelihoods_y + bpp_likelihoods_cbcr}



    def compress(self, Y, Cb, Cr):
        Y = Y.permute(0, 3, 1, 2) # b x 64 x 32 x 32
        Y1, Y2, Y3, Y4 = self.split_Y(Y) # Y1 b x 64 x 16 x 16 
        CbCr = torch.cat((Cb, Cr), dim=1) # b x 2 x 256 x 128
        z_cbcr_out = self.hyper_cbcr.compress(CbCr)  # string, shape, params
        h_cbcr = z_cbcr_out["params"] # b x 288 x 128 x 64
        CbCr_anchor, CbCr_non_anchor = self.split_CbCr(CbCr) # b x 4 x 128 x 64
        cbcr_anchor_out = self.Guassian_cbcr_anchor.compress(CbCr_anchor, h_cbcr) # string, shape, y_hat
        cbcr_non_anchor_out = self.Guassian_cbcr_non_anchor.compress(CbCr_non_anchor, torch.cat((h_cbcr, CbCr_anchor), dim=1))

        z_y_out = self.hyper_Y.compress(torch.cat((Y1, Y2, Y3, Y4), dim=1))# string, shape, params
        h_y = z_y_out["params"] # b x 192 x 16 x 16
        Y1_f = Y1.split(self.frequency, dim=1)
        y1_outs = []
        for i, f in enumerate(self.frequency):
            if i == 0:
                y1_out = self.Gaussion_Ys[i].compress(Y1_f[i], h_y)
                
            else:
                y1_out = self.Gaussion_Ys[i].compress(Y1_f[i], torch.cat((h_y, *Y1_f[:i]), dim=1))
            y1_outs.append(y1_out)

        prior_input = torch.cat((h_y, Y1), dim=1)  # b x 192+64 x 16 x 16
        prior_output = self.entropy_aprameters_prior(prior_input) # b x 192 x 16 x 16
        Y2_f = Y2.split(self.frequency, dim=1)
        Y3_f = Y3.split(self.frequency, dim=1)
        Y4_f = Y4.split(self.frequency, dim=1)
        y234_outs = []
        for i, f in enumerate(self.frequency):
            if i == 0:
                y234_out = self.Gaussion_Ys_234[i].compress(torch.cat((Y2_f[i], Y3_f[i], Y4_f[i]), dim=1), prior_output)
                
            else:
                y234_out = self.Gaussion_Ys_234[i].compress(torch.cat((Y2_f[i], Y3_f[i], Y4_f[i]), dim=1), 
                                                torch.cat((prior_output, *Y2_f[:i], *Y3_f[:i], *Y4_f[:i]), dim=1))
            y234_outs.append(y234_out)
        
        return {
            "z_cbcr": z_cbcr_out,
            "z_y": z_y_out,
            "cbcr_anchor": cbcr_anchor_out,
            "cbcr_non_anchor": cbcr_non_anchor_out,
            "y1": y1_outs,
            "y234": y234_outs,
        }

    def decompress(self, out_enc):
        z_cbcr = out_enc["z_cbcr"]
        z_y = out_enc["z_y"]
        cbcr_anchor_out = out_enc["cbcr_anchor"]
        cbcr_non_anchor_out = out_enc["cbcr_non_anchor"]
        y1_outs = out_enc["y1"]
        y234_outs = out_enc["y234"]
        z_CbCr = self.hyper_cbcr.decompress(z_cbcr['strings'], z_cbcr['shape'])
        cbcr_anchor_out = self.Guassian_cbcr_anchor.decompress(cbcr_anchor_out['strings'], cbcr_anchor_out['shape'], z_CbCr['params']) # string, shape, y_hat
        cbcr_non_anchor_out = self.Guassian_cbcr_non_anchor.decompress(cbcr_non_anchor_out['strings'], cbcr_non_anchor_out['shape'], 
                                                                       torch.cat((z_CbCr['params'], cbcr_anchor_out['y_hat']), dim=1))
        z_y = self.hyper_Y.decompress(z_y['strings'], z_y['shape'])
        y1_hats = []
        for i, f in enumerate(self.frequency):
            if i == 0:
                y1_hat = self.Gaussion_Ys[i].decompress(y1_outs[i]['strings'], y1_outs[i]['shape'], z_y['params'])
                
            else:
                y1_hat = self.Gaussion_Ys[i].decompress(y1_outs[i]['strings'], y1_outs[i]['shape'], torch.cat((z_y['params'], *y1_hats[:i]), dim=1))
            y1_hats.append(y1_hat['y_hat'])

        y1_hats = torch.cat(y1_hats, dim=1)
        prior_input = torch.cat((z_y['params'], y1_hats), dim=1)  # b x 192+64 x 16 x 16
        prior_output = self.entropy_aprameters_prior(prior_input) # b x 192 x 16 x 16
        y2_hats = []
        y3_hats = []
        y4_hats = []
        for i, f in enumerate(self.frequency):
            if i == 0:
                y234_hat = self.Gaussion_Ys_234[i].decompress(y234_outs[i]['strings'], y234_outs[i]['shape'], prior_output)
                y2_hat, y3_hat, y4_hat = y234_hat['y_hat'].chunk(3, dim=1)
                y2_hats.append(y2_hat)
                y3_hats.append(y3_hat)
                y4_hats.append(y4_hat)
                
            else:
                y234_hat = self.Gaussion_Ys_234[i].decompress(y234_outs[i]['strings'], y234_outs[i]['shape'], 
                                                torch.cat((prior_output, *y2_hats[:i], *y3_hats[:i], *y4_hats[:i]), dim=1))
                y2_hat, y3_hat, y4_hat = y234_hat['y_hat'].chunk(3, dim=1)
                y2_hats.append(y2_hat)
                y3_hats.append(y3_hat)
                y4_hats.append(y4_hat)

        y2_hats = torch.cat(y2_hats, dim=1)
        y3_hats = torch.cat(y3_hats, dim=1)
        y4_hats = torch.cat(y4_hats, dim=1)
        Y_hats = self.merge_Y(y1_hats, y2_hats, y3_hats, y4_hats)
        CbCr_hats = self.merge_CbCr(cbcr_anchor_out['y_hat'], cbcr_non_anchor_out['y_hat'])
        Cb_hats, Cr_hats = torch.chunk(CbCr_hats, 2, dim=1)
        return Y_hats.permute(0, 2, 3, 1).int(), Cb_hats.int(), Cr_hats.int()
            

        


        


    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net

@register_model("elic2022-official")
class Elic2022Official(SimpleVAECompressionModel):
    """ELIC 2022; uneven channel groups with checkerboard spatial context.

    Context model from [He2022].
    Based on modified attention model architecture from [Cheng2020].

    [He2022]: `"ELIC: Efficient Learned Image Compression with
    Unevenly Grouped Space-Channel Contextual Adaptive Coding"
    <https://arxiv.org/abs/2203.10886>`_, by Dailan He, Ziming Yang,
    Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang, CVPR 2022.

    [Cheng2020]: `"Learned Image Compression with Discretized Gaussian
    Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun,
    Masaru Takeuchi, and Jiro Katto, CVPR 2020.

    Args:
        N (int): Number of main network channels
        M (int): Number of latent space channels
        groups (list[int]): Number of channels in each channel group
    """

    def __init__(self, N=192, M=320, groups=None, **kwargs):
        super().__init__(**kwargs)

        if groups is None:
            groups = [16, 16, 32, 64, M - 128]

        self.groups = list(groups)
        assert sum(self.groups) == M

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            AttentionBlock(N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, M, kernel_size=5, stride=2),
            AttentionBlock(M),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            AttentionBlock(N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
        )

        h_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N, N * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N * 3 // 2, N * 2, kernel_size=3, stride=1),
        )

        # In [He2022], this is labeled "g_ch^(k)".
        channel_context = {
            f"y{k}": sequential_channel_ramp(
                sum(self.groups[:k]),
                self.groups[k] * 2,
                min_ch=N,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(1, len(self.groups))
        }

        # In [He2022], this is labeled "g_sp^(k)".
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + N * 2,
                self.groups[k] * 2,
                min_ch=N * 2,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # [He2022] uses a "hyperprior" architecture, which reconstructs y using z.
        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                # Channel groups with space-channel context model (SCCTX):
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                # Side information branch containing z:
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


@register_model("elic2022-chandelier")
class Elic2022Chandelier(SimpleVAECompressionModel):
    """ELIC 2022; simplified context model using only first and most recent groups.

    Context model from [He2022], with simplifications and parameters
    from the [Chandelier2023] implementation.
    Based on modified attention model architecture from [Cheng2020].

    .. note::

        This implementation contains some differences compared to the
        original [He2022] paper. For instance, the implemented context
        model only uses the first and the most recently decoded channel
        groups to predict the current channel group. In contrast, the
        original paper uses all previously decoded channel groups.
        Also, the last layer of h_s is now a conv rather than a deconv.

    [Chandelier2023]: `"ELiC-ReImplemetation"
    <https://github.com/VincentChandelier/ELiC-ReImplemetation>`_, by
    Vincent Chandelier, 2023.

    [He2022]: `"ELIC: Efficient Learned Image Compression with
    Unevenly Grouped Space-Channel Contextual Adaptive Coding"
    <https://arxiv.org/abs/2203.10886>`_, by Dailan He, Ziming Yang,
    Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang, CVPR 2022.

    [Cheng2020]: `"Learned Image Compression with Discretized Gaussian
    Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun,
    Masaru Takeuchi, and Jiro Katto, CVPR 2020.

    Args:
        N (int): Number of main network channels
        M (int): Number of latent space channels
        groups (list[int]): Number of channels in each channel group
    """

    def __init__(self, N=192, M=320, groups=None, **kwargs):
        super().__init__(**kwargs)

        if groups is None:
            groups = [16, 16, 32, 64, M - 128]

        self.groups = list(groups)
        assert sum(self.groups) == M

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            AttentionBlock(N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, M, kernel_size=5, stride=2),
            AttentionBlock(M),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            AttentionBlock(N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
        )

        h_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N, N * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(N * 3 // 2, M * 2, kernel_size=3, stride=1),
        )

        # In [He2022], this is labeled "g_ch^(k)".
        channel_context = {
            f"y{k}": nn.Sequential(
                conv(
                    # Input: first group, and most recently decoded group.
                    self.groups[0] + (k > 1) * self.groups[k - 1],
                    224,
                    kernel_size=5,
                    stride=1,
                ),
                nn.ReLU(inplace=True),
                conv(224, 128, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                conv(128, self.groups[k] * 2, kernel_size=5, stride=1),
            )
            for k in range(1, len(self.groups))
        }

        # In [He2022], this is labeled "g_sp^(k)".
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            nn.Sequential(
                conv1x1(
                    # Input: spatial context, channel context, and hyper params.
                    self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + M * 2,
                    M * 2,
                ),
                nn.ReLU(inplace=True),
                conv1x1(M * 2, 512),
                nn.ReLU(inplace=True),
                conv1x1(512, self.groups[k] * 2),
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(
                        quantizer="ste", chunks=("means", "scales")
                    ),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # [He2022] uses a "hyperprior" architecture, which reconstructs y using z.
        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                # Channel groups with space-channel context model (SCCTX):
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                # Side information branch containing z:
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )

        self._monkey_patch()

    def _monkey_patch(self):
        """Monkey-patch to use only first group and most recent group."""

        def merge_y(self: ChannelGroupsLatentCodec, *args):
            if len(args) == 0:
                return Tensor()
            if len(args) == 1:
                return args[0]
            if len(args) < len(self.groups):
                return torch.cat([args[0], args[-1]], dim=1)
            return torch.cat(args, dim=1)

        chan_groups_latent_codec = self.latent_codec["y"]
        obj = chan_groups_latent_codec
        obj.merge_y = types.MethodType(merge_y, obj)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


class ResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block.

    Introduced by [He2016], this block sandwiches a 3x3 convolution
    between two 1x1 convolutions which reduce and then restore the
    number of channels. This reduces the number of parameters required.

    [He2016]: `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/abs/1512.03385>`_, by Kaiming He, Xiangyu Zhang,
    Shaoqing Ren, and Jian Sun, CVPR 2016.

    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = min(in_ch, out_ch) // 2
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_ch, mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(mid_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)

        out = x
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out + identity
