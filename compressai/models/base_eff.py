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

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    GaussianConditionalLatentCodec,
    GaussianConditionalLatentCodec_ST,
    LaplaceConditionalLatentCodec_ST,
    GMMConditionalLatentCodec_ST,
    HyperLatentCodec,
)
from itertools import accumulate

from .base import  CompressionModel
from .utils import conv, deconv

class EfficientJPEGRecompression(CompressionModel):
    """Efficient JPEG recompression model.
    Args:
        N (int): Number of channels
        chunk (tuple): chunk type  such as ("scales", "means")
    """

    def __init__(self, N=192, M=288, chunk=("scales", "means"), **kwargs):
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
    
    # def bpp_loss(self, likelihoods):
    #     num_pixels = likelihoods.size(0) * likelihoods.size(2) * likelihoods.size(3)
    #     return torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    def bpp_loss(self, likelihoods):
        return torch.log(likelihoods).sum()

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


#########################################################################################################################################
from itertools import repeat
from typing import Callable, Optional

def to_2tuple(x):
    return tuple(repeat(x, 2))


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, bias=True
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class PatchUnEmbed_CbCr(nn.Module):
    """Patch Embedding to 2D Image."""

    def __init__(
        self, img_size=224, patch_size=16, out_chans=3, embed_dim=768
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(embed_dim, out_chans, kernel_size=1, stride=1)
        self.upconv = nn.Sequential(
            deconv(embed_dim, embed_dim, stride=2, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            deconv(embed_dim, embed_dim, stride=2, kernel_size=3)
        )

    def forward(self, x):
        B, N, C = x.shape  # B, 64, 256
        x = x.contiguous().view(B, self.grid_size[0], self.grid_size[1], C)  
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.upconv(x)
        x = self.proj(x)
        return x
    
class PatchUnEmbed_Y(nn.Module):
    """Patch Embedding to 2D Image."""

    def __init__(
        self, img_size=224, patch_size=16, out_chans=3, embed_dim=768
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(embed_dim, out_chans, kernel_size=1, stride=1)

    def forward(self, x):
        B, N, C = x.shape  # B, 64, 256
        x = x.contiguous().view(B, self.grid_size[0], self.grid_size[1], C)  
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale=True,
        with_cp=False,
        ffn_layer=Mlp,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.norm1 = norm_layer(dim)
        
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma1 = nn.Parameter(torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        def _inner_forward(x):
            if self.layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

       
        x = _inner_forward(x)

        return x

class EfficientJPEGRecompression_ViT(CompressionModel):
    """Efficient JPEG recompression model.
    Args:
        N (int): Number of channels
        chunk (tuple): chunk type  such as ("scales", "means")
    """

    def __init__(self, N=192, M=288, chunk=("scales", "means"), **kwargs):
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

            PatchEmbed(img_size=64, patch_size=4, in_chans=M, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_CbCr(img_size=64, patch_size=4, out_chans=len(self.chunk)*4, embed_dim=256),
        )

        entropy_parameters_CbCr_non_anchor = nn.Sequential(
            PatchEmbed(img_size=64, patch_size=4, in_chans=M+4, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_CbCr(img_size=64, patch_size=4, out_chans=len(self.chunk)*4, embed_dim=256),
        )

        self.entropy_aprameters_prior = nn.Sequential(
            PatchEmbed(img_size=16, patch_size=1, in_chans=N+64, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_Y(img_size=16, patch_size=1, out_chans=N, embed_dim=256),
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
            self._make_Gaussian_entropy_module(i, f) for i,  f in zip(in_channels_Y1, self.frequency)
        ])
        self.Gaussion_Ys_234 = nn.ModuleList([
            self._make_Gaussian_entropy_module(i, 3*f) for i,  f in zip(in_channels_Y234, self.frequency)
        ])


        self.Guassian_cbcr_anchor = self.gaussian_latent_encode(chunks=self.chunk,
                                                                   entropy_parameters=entropy_parameters_CbCr_anchor)
        self.Guassian_cbcr_non_anchor = self.gaussian_latent_encode(chunks=self.chunk,
                                                                       entropy_parameters=entropy_parameters_CbCr_non_anchor)
    
    def _make_Gaussian_entropy_module(self, in_channels, out_channels):
        entropy_aprameters = nn.Sequential(
            # conv(in_channels, channel, kernel_size=1, stride=1),
            # conv(channel, channel, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            # conv(channel, channel, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            # conv(channel, len(self.chunk)*out_channels, kernel_size=3, stride=1),
            PatchEmbed(img_size=16, patch_size=1, in_chans=in_channels, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_Y(img_size=16, patch_size=1, out_chans=len(self.chunk)*out_channels, embed_dim=256),
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
    
    # def bpp_loss(self, likelihoods):
    #     num_pixels = likelihoods.size(0) * likelihoods.size(2) * likelihoods.size(3)
    #     return torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    def bpp_loss(self, likelihoods):
        return torch.log(likelihoods).sum()

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
    

class EfficientJPEGRecompression_ViT_GMM(CompressionModel):
    """Efficient JPEG recompression model.
    Args:
        N (int): Number of channels
        chunk (tuple): chunk type  such as ("scales1", "scales2", "scales3", "means", "weights1", "weights2", "weights3")
    """

    def __init__(self, N=192, M=288, chunk=("scales1", "scales2", "scales3", "means1", "means2", "means3", "weights1", "weights2", "weights3"), **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.M = M
        self.chunk = chunk

        self.frequency = [28, 8, 7, 6, 5, 4, 3, 2, 1]
        cumulative_sum = list(accumulate(self.frequency, initial=0))  # [0, 28, 36, 43, 49, 54, 58, 61, 63, 64]
        in_channels_Y1 = [N + sum_val for sum_val in cumulative_sum[:-1]]  # 去掉最后一个元素
        in_channels_Y234 = [N + 3*sum_val for sum_val in cumulative_sum[:-1]]  # 去掉最后一个元素

        self.gaussian_latent_encode = GMMConditionalLatentCodec_ST

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

            PatchEmbed(img_size=64, patch_size=4, in_chans=M, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_CbCr(img_size=64, patch_size=4, out_chans=len(self.chunk)*4, embed_dim=256),
        )

        entropy_parameters_CbCr_non_anchor = nn.Sequential(
            PatchEmbed(img_size=64, patch_size=4, in_chans=M+4, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_CbCr(img_size=64, patch_size=4, out_chans=len(self.chunk)*4, embed_dim=256),
        )

        self.entropy_aprameters_prior = nn.Sequential(
            PatchEmbed(img_size=16, patch_size=1, in_chans=N+64, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_Y(img_size=16, patch_size=1, out_chans=N, embed_dim=256),
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
            self._make_Gaussian_entropy_module(i, f) for i,  f in zip(in_channels_Y1, self.frequency)
        ])
        self.Gaussion_Ys_234 = nn.ModuleList([
            self._make_Gaussian_entropy_module(i, 3*f) for i,  f in zip(in_channels_Y234, self.frequency)
        ])


        self.Guassian_cbcr_anchor = self.gaussian_latent_encode(chunks=self.chunk,
                                                                   entropy_parameters=entropy_parameters_CbCr_anchor)
        self.Guassian_cbcr_non_anchor = self.gaussian_latent_encode(chunks=self.chunk,
                                                                       entropy_parameters=entropy_parameters_CbCr_non_anchor)
    
    def _make_Gaussian_entropy_module(self, in_channels, out_channels):
        entropy_aprameters = nn.Sequential(
            # conv(in_channels, channel, kernel_size=1, stride=1),
            # conv(channel, channel, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            # conv(channel, channel, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            # conv(channel, len(self.chunk)*out_channels, kernel_size=3, stride=1),
            PatchEmbed(img_size=16, patch_size=1, in_chans=in_channels, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_Y(img_size=16, patch_size=1, out_chans=len(self.chunk)*out_channels, embed_dim=256),
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
    
    # def bpp_loss(self, likelihoods):
    #     num_pixels = likelihoods.size(0) * likelihoods.size(2) * likelihoods.size(3)
    #     return torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    def bpp_loss(self, likelihoods):
        return torch.log(likelihoods).sum()

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


class EfficientJPEGRecompression_ViT_Small(CompressionModel):
    """Efficient JPEG recompression model.
    Args:
        N (int): Number of channels
        chunk (tuple): chunk type  such as ("scales", "means")
    """

    def __init__(self, N=192, M=288, chunk=("scales", "means"), **kwargs):
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

            PatchEmbed(img_size=64, patch_size=4, in_chans=M, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_CbCr(img_size=64, patch_size=4, out_chans=len(self.chunk)*4, embed_dim=256),
        )

        entropy_parameters_CbCr_non_anchor = nn.Sequential(
            PatchEmbed(img_size=64, patch_size=4, in_chans=M+4, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_CbCr(img_size=64, patch_size=4, out_chans=len(self.chunk)*4, embed_dim=256),
        )

        self.entropy_aprameters_prior = nn.Sequential(
            PatchEmbed(img_size=16, patch_size=1, in_chans=N+64, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_Y(img_size=16, patch_size=1, out_chans=N, embed_dim=256),
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
            self._make_Gaussian_entropy_module(i, f) for i,  f in zip(in_channels_Y1, self.frequency)
        ])
        self.Gaussion_Ys_234 = nn.ModuleList([
            self._make_Gaussian_entropy_module(i, 3*f) for i,  f in zip(in_channels_Y234, self.frequency)
        ])


        self.Guassian_cbcr_anchor = self.gaussian_latent_encode(chunks=self.chunk,
                                                                   entropy_parameters=entropy_parameters_CbCr_anchor)
        self.Guassian_cbcr_non_anchor = self.gaussian_latent_encode(chunks=self.chunk,
                                                                       entropy_parameters=entropy_parameters_CbCr_non_anchor)
    
    def _make_Gaussian_entropy_module(self, in_channels, out_channels):
        entropy_aprameters = nn.Sequential(
            # conv(in_channels, channel, kernel_size=1, stride=1),
            # conv(channel, channel, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            # conv(channel, channel, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            # conv(channel, len(self.chunk)*out_channels, kernel_size=3, stride=1),
            PatchEmbed(img_size=16, patch_size=1, in_chans=in_channels, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_Y(img_size=16, patch_size=1, out_chans=len(self.chunk)*out_channels, embed_dim=256),
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
    
    # def bpp_loss(self, likelihoods):
    #     num_pixels = likelihoods.size(0) * likelihoods.size(2) * likelihoods.size(3)
    #     return torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    def bpp_loss(self, likelihoods):
        return torch.log(likelihoods).sum()

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
    
class EfficientJPEGRecompression_ViT_Meduim(CompressionModel):
    """Efficient JPEG recompression model.
    Args:
        N (int): Number of channels
        chunk (tuple): chunk type  such as ("scales", "means")
    """

    def __init__(self, N=192, M=288, chunk=("scales", "means"), **kwargs):
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

            PatchEmbed(img_size=64, patch_size=4, in_chans=M, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            # Block(dim=256, num_heads=8, mlp_ratio=4),
            # Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_CbCr(img_size=64, patch_size=4, out_chans=len(self.chunk)*4, embed_dim=256),
        )

        entropy_parameters_CbCr_non_anchor = nn.Sequential(
            PatchEmbed(img_size=64, patch_size=4, in_chans=M+4, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            # Block(dim=256, num_heads=8, mlp_ratio=4),
            # Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_CbCr(img_size=64, patch_size=4, out_chans=len(self.chunk)*4, embed_dim=256),
        )

        self.entropy_aprameters_prior = nn.Sequential(
            PatchEmbed(img_size=16, patch_size=1, in_chans=N+64, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            # Block(dim=256, num_heads=8, mlp_ratio=4),
            # Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_Y(img_size=16, patch_size=1, out_chans=N, embed_dim=256),
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
            self._make_Gaussian_entropy_module(i, f) for i,  f in zip(in_channels_Y1, self.frequency)
        ])
        self.Gaussion_Ys_234 = nn.ModuleList([
            self._make_Gaussian_entropy_module(i, 3*f) for i,  f in zip(in_channels_Y234, self.frequency)
        ])


        self.Guassian_cbcr_anchor = self.gaussian_latent_encode(chunks=self.chunk,
                                                                   entropy_parameters=entropy_parameters_CbCr_anchor)
        self.Guassian_cbcr_non_anchor = self.gaussian_latent_encode(chunks=self.chunk,
                                                                       entropy_parameters=entropy_parameters_CbCr_non_anchor)
    
    def _make_Gaussian_entropy_module(self, in_channels, out_channels):
        entropy_aprameters = nn.Sequential(
            # conv(in_channels, channel, kernel_size=1, stride=1),
            # conv(channel, channel, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            # conv(channel, channel, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            # conv(channel, len(self.chunk)*out_channels, kernel_size=3, stride=1),
            PatchEmbed(img_size=16, patch_size=1, in_chans=in_channels, embed_dim=256),
            Block(dim=256, num_heads=8, mlp_ratio=4),
            # Block(dim=256, num_heads=8, mlp_ratio=4),
            # Block(dim=256, num_heads=8, mlp_ratio=4),
            PatchUnEmbed_Y(img_size=16, patch_size=1, out_chans=len(self.chunk)*out_channels, embed_dim=256),
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
    
    # def bpp_loss(self, likelihoods):
    #     num_pixels = likelihoods.size(0) * likelihoods.size(2) * likelihoods.size(3)
    #     return torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    def bpp_loss(self, likelihoods):
        return torch.log(likelihoods).sum()

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


###################Guo 422###############################################

class CFM(nn.Module):
    def __init__(self):
        super().__init__()  
        self.Y_encoder = nn.Sequential(
            conv(1, 128, stride=2, kernel_size=5),
            nn.PReLU(),)
        # self.CbCr_encoder = nn.Sequential(  # 444
        #     conv(2, 128, stride=2, kernel_size=5),
        #     nn.PReLU(),)

        self.CbCr_encoder = nn.Sequential(  #422
            conv(2, 128, stride=1, kernel_size=3),
            nn.PReLU(),)
        
        self.conv = conv(128*2, 192, stride=1, kernel_size=1)
        self.act = nn.PReLU()
    
    def forward(self, Y, Cb, Cr):
        Y = self.Y_encoder(Y)
        CbCr = self.CbCr_encoder(torch.cat((Cb, Cr), dim=1))
        YCbCr = torch.cat((Y, CbCr), dim=1)
        YCbCr = self.conv(YCbCr)
        YCbCr = self.act(YCbCr)
        return YCbCr


class CPSM(nn.Module):
    def __init__(self):
        super().__init__()  
        self.act = nn.PReLU()
        self.conv = conv(384, 384, stride=1, kernel_size=1)
        self.dconvcr = nn.Sequential(nn.PReLU(),
                                    deconv(128, 128, stride=1, kernel_size=3))
        self.dconvcb = nn.Sequential(nn.PReLU(),
                                    deconv(128, 128, stride=1, kernel_size=3))
        self.dconvY = nn.Sequential(nn.PReLU(),
                                    deconv(128, 128, stride=2, kernel_size=5))
        
    
    def forward(self, X):
        X = self.conv(self.act(X))
        Cr, Cb, Y = torch.chunk(X, 3, dim=1)
        Cb = self.dconvcb(Cb)
        Cr = self.dconvcr(Cr)
        Y = self.dconvY(Y)
        return Cr, Cb, Y
    

class GuoJPEGRecompression422(CompressionModel):
    """Guo DCT JPEG recompression model.
    Args:
        N (int): Number of channels
        chunk (tuple): chunk type  such as ("scales", "means")
    """

    def __init__(self, N=192, M=288, chunk=("scales", "means"), **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.M = M
        self.chunk = chunk

        self.frequency = [28, 8, 7, 6, 5, 4, 3, 2, 1]
        cumulative_sum = list(accumulate(self.frequency, initial=0))  # [0, 28, 36, 43, 49, 54, 58, 61, 63, 64]
        in_channels_Y1 = [N + sum_val for sum_val in cumulative_sum[:-1]]  # 去掉最后一个元素
        in_channels_Y234 = [N + 3*sum_val for sum_val in cumulative_sum[:-1]]  # 去掉最后一个元素

        self.gaussian_latent_encode = LaplaceConditionalLatentCodec_ST

        h_e = nn.Sequential(
            # conv(192, 192, stride=1, kernel_size=3),
            # nn.LeakyReLU(inplace=True),
            conv(192, 192, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(192, 192, stride=2, kernel_size=5),
        )

        h_d = nn.Sequential(
            # conv(192, 192, stride=1, kernel_size=3),
            # nn.LeakyReLU(inplace=True),
            deconv(192, 288, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(288, 384, stride=2, kernel_size=5),
        )

        entropy_parameters_Cr = nn.Sequential(
            conv(128, 128, kernel_size=1, stride=1),
            # conv(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            # conv(128, 128, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            conv(128, len(self.chunk), kernel_size=3, stride=1),
        )

        entropy_parameters_Cb = nn.Sequential(   
            conv(128+1, 128, kernel_size=1, stride=1),
            # conv(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            # conv(128, 128, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            conv(128, len(self.chunk), kernel_size=3, stride=1),
        )

        self.CbCr_prior = nn.Sequential(  #  422  128   
            conv(1+1+128, 192, kernel_size=3, stride=2),    # 64
            conv(192, 192, kernel_size=3, stride=2),    # 32
            nn.ReLU(inplace=True),
            conv(192, 192, kernel_size=3, stride=2),    # 16
            nn.ReLU(inplace=True),
            conv(192, 192, kernel_size=3, stride=1),
        )

        self.Y_conv = conv(128, 128, stride=2, kernel_size=3)

        self.entropy_aprameters_prior = nn.Sequential(
            conv(N+64, N, kernel_size=1, stride=1),
            # conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            # conv(N, N, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=1),
        )

        self.hyper = HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_e,
                    h_s=h_d,
                    quantizer="ste",
                )

        
        self.CFM = CFM()
        self.CPSM = CPSM()

        self.Gaussion_Ys = nn.ModuleList([
            self._make_Gaussian_entropy_module(i, f, channel=N) for i,  f in zip(in_channels_Y1, self.frequency)
        ])
        self.Gaussion_Ys_234 = nn.ModuleList([
            self._make_Gaussian_entropy_module(i, 3*f, channel=N) for i,  f in zip(in_channels_Y234, self.frequency)
        ])


        self.Guassian_cr = self.gaussian_latent_encode(chunks=self.chunk,
                                                                   entropy_parameters=entropy_parameters_Cr)
        self.Guassian_cb = self.gaussian_latent_encode(chunks=self.chunk,
                                                                       entropy_parameters=entropy_parameters_Cb)
    
    def _make_Gaussian_entropy_module(self, in_channels, out_channels, channel):
        entropy_aprameters = nn.Sequential(
            conv(in_channels, channel, kernel_size=1, stride=1),
            # conv(channel, channel, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            # conv(channel, channel, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
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
        return torch.log(likelihoods).sum()

    def forward(self, Y, Cb, Cr):  # Y b x 32 x 32 x 64
        bpp_likelihoods_z = 0
        bpp_likelihoods_y = 0
        bpp_likelihoods_cbcr = 0
        Y = Y.permute(0, 3, 1, 2) # b x 64 x 32 x 32
        Y_ori = Y.clone().reshape(Y.shape[0], 8, 8, Y.shape[2], Y.shape[3]) # b x 8 x 8 x 32 x 32
        Y_ori = Y_ori.permute(0, 1, 3, 2, 4).reshape(Y.shape[0], 1, 8*32, 8*32) # b x 1 x 256 x 256

        Y1, Y2, Y3, Y4 = self.split_Y(Y) # Y1 b x 64 x 16 x 16 
        cfm_feat = self.CFM(Y_ori, Cb, Cr)
        YCbCr_out = self.hyper(cfm_feat)
  
        bpp_likelihoods_z += self.bpp_loss(YCbCr_out["likelihoods"]["z"])
        h_YCbCr = YCbCr_out["params"] # b x 384 x 256 x 256
        Cr_prior, Cb_prior, Y_prior = self.CPSM(h_YCbCr)
        cr_out = self.Guassian_cr(Cr, Cr_prior)
        bpp_likelihoods_cbcr += self.bpp_loss(cr_out["likelihoods"]["y"])
       
        cb_out = self.Guassian_cb(Cb, torch.cat((Cr, Cb_prior), dim=1))
        bpp_likelihoods_cbcr += self.bpp_loss(cb_out["likelihoods"]["y"])

        h_y = self.CbCr_prior(torch.cat((Cb, Cr, self.Y_conv(Y_prior)), dim=1)) # b x 192 x 16 x 16
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
    



###################Guo 444###############################################

class CFM444(nn.Module):
    def __init__(self):
        super().__init__()  
        self.Y_encoder = nn.Sequential(
            conv(1, 128, stride=2, kernel_size=5),
            nn.PReLU(),)
        self.CbCr_encoder = nn.Sequential(  # 444
            conv(2, 128, stride=2, kernel_size=5),
            nn.PReLU(),)

        
        self.conv = conv(128*2, 192, stride=1, kernel_size=1)
        self.act = nn.PReLU()
    
    def forward(self, Y, Cb, Cr):
        Y = self.Y_encoder(Y)
        CbCr = self.CbCr_encoder(torch.cat((Cb, Cr), dim=1))
        YCbCr = torch.cat((Y, CbCr), dim=1)
        YCbCr = self.conv(YCbCr)
        YCbCr = self.act(YCbCr)
        return YCbCr


class CPSM444(nn.Module):
    def __init__(self):
        super().__init__()  
        self.act = nn.PReLU()
        self.conv = conv(384, 384, stride=1, kernel_size=1)
        self.dconvcr = nn.Sequential(nn.PReLU(),
                                    deconv(128, 128, stride=2, kernel_size=3))
        self.dconvcb = nn.Sequential(nn.PReLU(),
                                    deconv(128, 128, stride=2, kernel_size=3))
        self.dconvY = nn.Sequential(nn.PReLU(),
                                    deconv(128, 128, stride=2, kernel_size=5))
        
    
    def forward(self, X):
        X = self.conv(self.act(X))
        Cr, Cb, Y = torch.chunk(X, 3, dim=1)
        Cb = self.dconvcb(Cb)
        Cr = self.dconvcr(Cr)
        Y = self.dconvY(Y)
        return Cr, Cb, Y
    

class GuoJPEGRecompression444(CompressionModel):
    """Guo DCT JPEG recompression model.
    Args:
        N (int): Number of channels
        chunk (tuple): chunk type  such as ("scales", "means")
    """

    def __init__(self, N=192, M=288, chunk=("scales", "means"), **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.M = M
        self.chunk = chunk

        self.frequency = [28, 8, 7, 6, 5, 4, 3, 2, 1]
        cumulative_sum = list(accumulate(self.frequency, initial=0))  # [0, 28, 36, 43, 49, 54, 58, 61, 63, 64]
        in_channels_Y1 = [N + sum_val for sum_val in cumulative_sum[:-1]]  # 去掉最后一个元素
        in_channels_Y234 = [N + 3*sum_val for sum_val in cumulative_sum[:-1]]  # 去掉最后一个元素

        self.gaussian_latent_encode = LaplaceConditionalLatentCodec_ST

        h_e = nn.Sequential(
            # conv(192, 192, stride=1, kernel_size=3),
            # nn.LeakyReLU(inplace=True),
            conv(192, 192, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(192, 192, stride=2, kernel_size=5),
        )

        h_d = nn.Sequential(
            # conv(192, 192, stride=1, kernel_size=3),
            # nn.LeakyReLU(inplace=True),
            deconv(192, 288, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(288, 384, stride=2, kernel_size=5),
        )

        entropy_parameters_Cr = nn.Sequential(
            # conv(128, 128, kernel_size=1, stride=1),
            conv(128, 128, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            # conv(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(128, len(self.chunk), kernel_size=3, stride=1),
        )

        entropy_parameters_Cb = nn.Sequential(   
            conv(128+1, 128, kernel_size=1, stride=1),
            # conv(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            # conv(128, 128, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            conv(128, len(self.chunk), kernel_size=3, stride=1),
        )

        self.CbCr_prior = nn.Sequential(  #  444  256   
            conv(1+1+128, 192, kernel_size=3, stride=2),    # 128
            conv(192, 192, kernel_size=3, stride=2),    # 64
            nn.ReLU(inplace=True),
            conv(192, 192, kernel_size=3, stride=2),    # 32
            nn.ReLU(inplace=True),
            conv(192, 192, kernel_size=3, stride=2),    # 16
            # nn.ReLU(inplace=True),
            # conv(192, 192, kernel_size=3, stride=1),
        )


        self.entropy_aprameters_prior = nn.Sequential(
            conv(N+64, N, kernel_size=1, stride=1),
            # conv(N, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            # conv(N, N, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=1),
        )

        self.hyper = HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_e,
                    h_s=h_d,
                    quantizer="ste",
                )

        
        self.CFM = CFM444()
        self.CPSM = CPSM444()

        self.Gaussion_Ys = nn.ModuleList([
            self._make_Gaussian_entropy_module(i, f, channel=N) for i,  f in zip(in_channels_Y1, self.frequency)
        ])
        self.Gaussion_Ys_234 = nn.ModuleList([
            self._make_Gaussian_entropy_module(i, 3*f, channel=N) for i,  f in zip(in_channels_Y234, self.frequency)
        ])


        self.Guassian_cr = self.gaussian_latent_encode(chunks=self.chunk,
                                                                   entropy_parameters=entropy_parameters_Cr)
        self.Guassian_cb = self.gaussian_latent_encode(chunks=self.chunk,
                                                                       entropy_parameters=entropy_parameters_Cb)
    
    def _make_Gaussian_entropy_module(self, in_channels, out_channels, channel):
        entropy_aprameters = nn.Sequential(
            conv(in_channels, 64, kernel_size=1, stride=1),
            # conv(channel, channel, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            # conv(channel, channel, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            conv(64, len(self.chunk)*out_channels, kernel_size=3, stride=1),
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
        return torch.log(likelihoods).sum()

    def forward(self, Y, Cb, Cr):  # Y b x 32 x 32 x 64
        bpp_likelihoods_z = 0
        bpp_likelihoods_y = 0
        bpp_likelihoods_cbcr = 0
        Y = Y.permute(0, 3, 1, 2) # b x 64 x 32 x 32
        Y_ori = Y.clone().reshape(Y.shape[0], 8, 8, Y.shape[2], Y.shape[3]) # b x 8 x 8 x 32 x 32
        Y_ori = Y_ori.permute(0, 1, 3, 2, 4).reshape(Y.shape[0], 1, 8*32, 8*32) # b x 1 x 256 x 256

        Y1, Y2, Y3, Y4 = self.split_Y(Y) # Y1 b x 64 x 16 x 16 
        cfm_feat = self.CFM(Y_ori, Cb, Cr)
        YCbCr_out = self.hyper(cfm_feat)
  
        bpp_likelihoods_z += self.bpp_loss(YCbCr_out["likelihoods"]["z"])
        h_YCbCr = YCbCr_out["params"] # b x 384 x 256 x 256
        Cr_prior, Cb_prior, Y_prior = self.CPSM(h_YCbCr)
        cr_out = self.Guassian_cr(Cr, Cr_prior)
        bpp_likelihoods_cbcr += self.bpp_loss(cr_out["likelihoods"]["y"])
       
        cb_out = self.Guassian_cb(Cb, torch.cat((Cr, Cb_prior), dim=1))
        bpp_likelihoods_cbcr += self.bpp_loss(cb_out["likelihoods"]["y"])

        h_y = self.CbCr_prior(torch.cat((Cb, Cr, Y_prior), dim=1)) # b x 192 x 16 x 16
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

