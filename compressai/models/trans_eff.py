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
    GaussianConditionalLatentCodec_ST_Trans,
    PolynomialLaplaceConditionalLatentCodec_ST,
    GaussianStudentConditionalLatentCodec_ST,
    PolynomialLaplaceConditionalLatentCodec_ST2,
    HyperLatentCodec,
)
from itertools import accumulate

from .base import  CompressionModel
from .utils import conv, deconv
from .trans_network import FractalGen


#### base
def fractalar_base_in32(rle, gaussian, **kwargs):  #small
    if gaussian:
        channel = 2
    else:
        channel = 3
    model = FractalGen(
        img_size_list=(32, 16, 4, 1),
            embed_dim_list=(768, 384, 192, 64),
            num_blocks_list=(6, 6, 3, 1), #1
            num_heads_list=(6, 6, 3, 4),
            generator_type_list=("ar", "ar", "ar", "ar"),
            fractal_level=0,
            rle=rle,
            channel=channel,
        **kwargs)
    return model


def fractalar_base_in16(rle, gaussian, **kwargs):
    if gaussian:
        channel = 2
    else:
        channel = 3
    model = FractalGen(
        img_size_list=(16, 4, 1),
            embed_dim_list=(384, 192, 64), ##(384, 192, 64)
            num_blocks_list=(6, 3, 1), ##(6, 3, 1)
            num_heads_list=(6, 3, 4), ##(6, 3, 4)
            generator_type_list=("ar", "ar", "ar"),
            fractal_level=0,
            rle=rle,
            channel=channel,
        **kwargs)
    return model



try:
    import rle_cuda
except ImportError:
    print("CUDA extension not found. Compiling from source...")



####student-t

class TransJPEGRecompression_t(CompressionModel):
    """Efficient JPEG recompression model.
    Args:
        N (int): Number of channels
        chunk (tuple): chunk type  such as ("scales", "means")
    """


    def __init__(self, N=192, M=288, chunk=("scales", "means"), lsize=4, net='B', rle=False, gaussian=False, **kwargs): # lsize 4 (256)  12 (768)
        super().__init__(**kwargs)
        self.N = N
        self.M = M
        self.chunk = chunk
        self.lsize = lsize
        self.rle = rle
        self.gaussian = gaussian
        if net == 'B':
            self.entropy_net_Y = fractalar_base_in32(rle=rle, gaussian=gaussian)
            self.entropy_net_CbCr = fractalar_base_in16(rle=rle, gaussian=gaussian)
            y_cond_emb_dim = 768
            cbcr_cond_emb_dim = 384
        # elif net == 'L':
        #     self.entropy_net_Y = fractalar_large_in32(rle=rle, gaussian=gaussian)
        #     self.entropy_net_CbCr = fractalar_large_in16(rle=rle, gaussian=gaussian)
        #     y_cond_emb_dim = 1024
        #     cbcr_cond_emb_dim = 512
        
        # elif net == 'H':
        #     self.entropy_net_Y = fractalar_huge_in32(rle=rle, gaussian=gaussian)
        #     self.entropy_net_CbCr = fractalar_huge_in16(rle=rle, gaussian=gaussian)
        #     y_cond_emb_dim = 1280
        #     cbcr_cond_emb_dim = 640

        if gaussian:
            self.gaussian_latent_encode = GaussianConditionalLatentCodec_ST_Trans
            print('using gaussian latent code')
        else:
            self.gaussian_latent_encode = GaussianStudentConditionalLatentCodec_ST
            print('using laplace latent code')

        if rle:
            h_e_Y = nn.Sequential(
                conv(2*64*4, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )
        else:
            h_e_Y = nn.Sequential(
                conv(64*4, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )

        h_d_Y = nn.Sequential(
            conv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M, N, stride=1, kernel_size=3),
        )

        # if rle:
        #     h_e_Y = nn.Sequential(
        #         conv(2*64*4, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
                
        #     )
        # else:
        #     h_e_Y = nn.Sequential(
        #         conv(64*4, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #     )

        # h_d_Y = nn.Sequential(
        #     conv(N, N, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(N, M, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(M, M, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(M, N, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(N, N, stride=1, kernel_size=3),
        # )

        self.Y_cond_emb = nn.Linear(lsize*lsize*N, y_cond_emb_dim, bias=True)

        self.hyper_Y= HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_e_Y,
                    h_s=h_d_Y,
                    quantizer="ste",
                )

        if rle:
            h_e_C = nn.Sequential(
                conv(2*64, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )
        else:
            h_e_C = nn.Sequential(
                conv(64, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )

        h_d_C = nn.Sequential(
            conv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M, N, stride=1, kernel_size=3),
        )
        
        # if rle:
        #     h_e_C = nn.Sequential(
        #         conv(2*64, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #     )
        # else:
        #     h_e_C = nn.Sequential(
        #         conv(64, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #     )

        # h_d_C = nn.Sequential(
        #     conv(N, N, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(N, M, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(M, M, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(M, N, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(N, N, stride=1, kernel_size=3),
        # )

        
        self.CbCr_cond_emb = nn.Linear(lsize*lsize*N, cbcr_cond_emb_dim, bias=True)

        self.hyper_cbcr= HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_e_C,
                    h_s=h_d_C,
                    quantizer="ste",
                )
        
        self.Gaussian_Y = self.gaussian_latent_encode(chunks=self.chunk,  entropy_parameters=self.entropy_net_Y)
        self.Gaussian_CbCr = self.gaussian_latent_encode(chunks=self.chunk,  entropy_parameters=self.entropy_net_CbCr)

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
    

    def forward(self, Y, Cb, Cr): 
        bpp_likelihoods_z = 0
        bpp_likelihoods_y = 0
        bpp_likelihoods_cbcr = 0
        regs = 0
        CbCr = torch.cat((Cb, Cr), dim=0) # parallel 
        if self.rle:
            b, h, w, _ = Y.shape
            Y = Y.reshape(-1, 64)
            Y_ori = Y.clone()
            Y_ori = Y_ori.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            Y, Ymasks = rle_cuda.rle_encode(Y.clone().long(), 3)
            # Y_n, Ymasks_n = self.batch_rle_encode_full_expand(Y.clone().long(), max_segments=3)
            Y = Y.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            Ymasks = Ymasks.reshape(b, h, w, 64).permute(0, 3, 1, 2)

            b, h, w, _ = CbCr.shape
            CbCr = CbCr.reshape(-1, 64)
            CbCr_ori = CbCr.clone()
            CbCr_ori = CbCr_ori.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            CbCr, CbCrmasks = rle_cuda.rle_encode(CbCr.clone().long(), 3)
            # CbCr_n, CbCrmasks_n = self.batch_rle_encode_full_expand(CbCr.clone().long(), max_segments=3)
            CbCr = CbCr.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            CbCrmasks = CbCrmasks.reshape(b, h, w, 64).permute(0, 3, 1, 2)
            Y1, Y2, Y3, Y4 = self.split_Y(Y) 
            Y1_ori, Y2_ori, Y3_ori, Y4_ori = self.split_Y(Y_ori) 
            y_out = self.hyper_Y(torch.cat((Y1, Y2, Y3, Y4, Y1_ori, Y2_ori, Y3_ori, Y4_ori), dim=1))
        else:
            Y = Y.permute(0, 3, 1, 2) 
            CbCr = CbCr.permute(0, 3, 1, 2)
            
            Y1, Y2, Y3, Y4 = self.split_Y(Y) 
            y_out = self.hyper_Y(torch.cat((Y1, Y2, Y3, Y4), dim=1))
        z_y_likelihoods = y_out["likelihoods"]["z"]
        bpp_likelihoods_z += self.bpp_loss(z_y_likelihoods)  # z
        h_y = y_out["params"] 
        cond_Y = self.Y_cond_emb(h_y.reshape(-1, self.lsize*self.lsize*self.N))
        if self.rle:
            y_out = self.Gaussian_Y(Y, cond_Y, Ymasks)
        else:
            y_out = self.Gaussian_Y(Y, cond_Y, None)
            
        bpp_likelihoods_y += self.bpp_loss(y_out["likelihoods"]["y"]) # y

        if self.rle:
            cbcr_out = self.hyper_cbcr(torch.cat((CbCr, CbCr_ori), dim=1)) 
        else:
            cbcr_out = self.hyper_cbcr(CbCr)
        z_cbcr_likelihoods = cbcr_out["likelihoods"]["z"]
        bpp_likelihoods_z += self.bpp_loss(z_cbcr_likelihoods)  # z
        h_cbcr = cbcr_out["params"] 
        cond_CbCr = self.CbCr_cond_emb(h_cbcr.reshape(-1, self.lsize*self.lsize*self.N))

        if self.rle:
            CbCr_out = self.Gaussian_CbCr(CbCr, cond_CbCr, CbCrmasks)
        else:
            CbCr_out = self.Gaussian_CbCr(CbCr, cond_CbCr, None)
        bpp_likelihoods_cbcr += self.bpp_loss(CbCr_out["likelihoods"]["y"]) # y
        regs += CbCr_out["reg"]
        regs += y_out["reg"]
        return {"bpp_loss": bpp_likelihoods_z + bpp_likelihoods_y + bpp_likelihoods_cbcr, "reg": regs}
        



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



class TransJPEGRecompression2(CompressionModel):
    """Efficient JPEG recompression model.
    Args:
        N (int): Number of channels
        chunk (tuple): chunk type  such as ("scales", "means")
    """


    def __init__(self, N=192, M=288, chunk=("scales", "means"), lsize=4, net='B', rle=False, gaussian=False, **kwargs): # lsize 4 (256)  12 (768)
        super().__init__(**kwargs)
        self.N = N
        self.M = M
        self.chunk = chunk
        self.lsize = lsize
        self.rle = rle
        self.gaussian = gaussian
        if net == 'B':
            self.entropy_net_Y = fractalar_base_in32(rle=rle, gaussian=gaussian)
            self.entropy_net_CbCr = fractalar_base_in16(rle=rle, gaussian=gaussian)
            y_cond_emb_dim = 768
            cbcr_cond_emb_dim = 384
        # elif net == 'L':
        #     self.entropy_net_Y = fractalar_large_in32(rle=rle, gaussian=gaussian)
        #     self.entropy_net_CbCr = fractalar_large_in16(rle=rle, gaussian=gaussian)
        #     y_cond_emb_dim = 1024
        #     cbcr_cond_emb_dim = 512
        
        # elif net == 'H':
        #     self.entropy_net_Y = fractalar_huge_in32(rle=rle, gaussian=gaussian)
        #     self.entropy_net_CbCr = fractalar_huge_in16(rle=rle, gaussian=gaussian)
        #     y_cond_emb_dim = 1280
        #     cbcr_cond_emb_dim = 640

        if gaussian:
            self.gaussian_latent_encode = GaussianConditionalLatentCodec_ST_Trans
            print('using gaussian latent code')
        else:
            self.gaussian_latent_encode = PolynomialLaplaceConditionalLatentCodec_ST2
            print('using laplace latent code')

        if rle:
            h_e_Y = nn.Sequential(
                conv(2*64*4, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )
        else:
            h_e_Y = nn.Sequential(
                conv(64*4, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )

        h_d_Y = nn.Sequential(
            conv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M, N, stride=1, kernel_size=3),
        )

        # if rle:
        #     h_e_Y = nn.Sequential(
        #         conv(2*64*4, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
                
        #     )
        # else:
        #     h_e_Y = nn.Sequential(
        #         conv(64*4, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #     )

        # h_d_Y = nn.Sequential(
        #     conv(N, N, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(N, M, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(M, M, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(M, N, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(N, N, stride=1, kernel_size=3),
        # )

        self.Y_cond_emb = nn.Linear(lsize*lsize*N, y_cond_emb_dim, bias=True)

        self.hyper_Y= HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_e_Y,
                    h_s=h_d_Y,
                    quantizer="ste",
                )

        if rle:
            h_e_C = nn.Sequential(
                conv(2*64, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )
        else:
            h_e_C = nn.Sequential(
                conv(64, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )

        h_d_C = nn.Sequential(
            conv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M, N, stride=1, kernel_size=3),
        )
        
        # if rle:
        #     h_e_C = nn.Sequential(
        #         conv(2*64, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #     )
        # else:
        #     h_e_C = nn.Sequential(
        #         conv(64, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #     )

        # h_d_C = nn.Sequential(
        #     conv(N, N, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(N, M, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(M, M, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(M, N, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(N, N, stride=1, kernel_size=3),
        # )

        
        self.CbCr_cond_emb = nn.Linear(lsize*lsize*N, cbcr_cond_emb_dim, bias=True)

        self.hyper_cbcr= HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_e_C,
                    h_s=h_d_C,
                    quantizer="ste",
                )
        
        self.Gaussian_Y = self.gaussian_latent_encode(chunks=self.chunk,  entropy_parameters=self.entropy_net_Y)
        self.Gaussian_CbCr = self.gaussian_latent_encode(chunks=self.chunk,  entropy_parameters=self.entropy_net_CbCr)

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
    

    def forward(self, Y, Cb, Cr): 
        bpp_likelihoods_z = 0
        bpp_likelihoods_y = 0
        bpp_likelihoods_cbcr = 0
        regs = 0
        CbCr = torch.cat((Cb, Cr), dim=0) # parallel 
        if self.rle:
            b, h, w, _ = Y.shape
            Y = Y.reshape(-1, 64)
            Y_ori = Y.clone()
            Y_ori = Y_ori.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            Y, Ymasks = rle_cuda.rle_encode(Y.clone().long(), 3)
            # Y_n, Ymasks_n = self.batch_rle_encode_full_expand(Y.clone().long(), max_segments=3)
            Y = Y.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            Ymasks = Ymasks.reshape(b, h, w, 64).permute(0, 3, 1, 2)

            b, h, w, _ = CbCr.shape
            CbCr = CbCr.reshape(-1, 64)
            CbCr_ori = CbCr.clone()
            CbCr_ori = CbCr_ori.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            CbCr, CbCrmasks = rle_cuda.rle_encode(CbCr.clone().long(), 3)
            # CbCr_n, CbCrmasks_n = self.batch_rle_encode_full_expand(CbCr.clone().long(), max_segments=3)
            CbCr = CbCr.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            CbCrmasks = CbCrmasks.reshape(b, h, w, 64).permute(0, 3, 1, 2)
            Y1, Y2, Y3, Y4 = self.split_Y(Y) 
            Y1_ori, Y2_ori, Y3_ori, Y4_ori = self.split_Y(Y_ori) 
            y_out = self.hyper_Y(torch.cat((Y1, Y2, Y3, Y4, Y1_ori, Y2_ori, Y3_ori, Y4_ori), dim=1))
        else:
            Y = Y.permute(0, 3, 1, 2) 
            CbCr = CbCr.permute(0, 3, 1, 2)
            
            Y1, Y2, Y3, Y4 = self.split_Y(Y) 
            y_out = self.hyper_Y(torch.cat((Y1, Y2, Y3, Y4), dim=1))
        z_y_likelihoods = y_out["likelihoods"]["z"]
        bpp_likelihoods_z += self.bpp_loss(z_y_likelihoods)  # z
        h_y = y_out["params"] 
        cond_Y = self.Y_cond_emb(h_y.reshape(-1, self.lsize*self.lsize*self.N))
        if self.rle:
            y_out = self.Gaussian_Y(Y, cond_Y, Ymasks)
        else:
            y_out = self.Gaussian_Y(Y, cond_Y, None)
            
        bpp_likelihoods_y += self.bpp_loss(y_out["likelihoods"]["y"]) # y

        if self.rle:
            cbcr_out = self.hyper_cbcr(torch.cat((CbCr, CbCr_ori), dim=1)) 
        else:
            cbcr_out = self.hyper_cbcr(CbCr)
        z_cbcr_likelihoods = cbcr_out["likelihoods"]["z"]
        bpp_likelihoods_z += self.bpp_loss(z_cbcr_likelihoods)  # z
        h_cbcr = cbcr_out["params"] 
        cond_CbCr = self.CbCr_cond_emb(h_cbcr.reshape(-1, self.lsize*self.lsize*self.N))

        if self.rle:
            CbCr_out = self.Gaussian_CbCr(CbCr, cond_CbCr, CbCrmasks)
        else:
            CbCr_out = self.Gaussian_CbCr(CbCr, cond_CbCr, None)
        bpp_likelihoods_cbcr += self.bpp_loss(CbCr_out["likelihoods"]["y"]) # y
        regs += CbCr_out["reg"]
        regs += y_out["reg"]
        return {"bpp_loss": bpp_likelihoods_z + bpp_likelihoods_y + bpp_likelihoods_cbcr, "reg": regs}
        



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



####################422
class TransJPEGRecompression422(CompressionModel):
    """Efficient JPEG recompression model.
    Args:
        N (int): Number of channels
        chunk (tuple): chunk type  such as ("scales", "means")
    """


    def __init__(self, N=192, M=288, chunk=("scales", "means"), lsize=4, net='B', rle=False, gaussian=False, **kwargs): # lsize 4 (256)  12 (768)
        super().__init__(**kwargs)
        self.N = N
        self.M = M
        self.chunk = chunk
        self.lsize = lsize
        self.rle = rle
        self.gaussian = gaussian
        if net == 'B':
            # from thop import profile
            # inputs = torch.randn(1, 768)
            # Y = torch.randn(1, 64, 32, 32)
            self.entropy_net_Y = fractalar_base_in32(rle=rle, gaussian=gaussian)   ###note AR last channel 3
            # flops, params = profile(self.entropy_net_Y, (Y, [inputs], None))
            # print('flops: ', flops, 'params: ', params)
            # inputs = torch.randn(1, 384)
            # CbCr = torch.randn(1, 64, 16, 16)
            self.entropy_net_CbCr = fractalar_base_in16(rle=rle, gaussian=gaussian)
            # flops, params = profile(self.entropy_net_CbCr, (CbCr, [inputs], None))
            # print('flops: ', flops, 'params: ', params)
            y_cond_emb_dim = 768
            cbcr_cond_emb_dim = 384
        # elif net == 'L':
        #     self.entropy_net_Y = fractalar_large_in32(rle=rle, gaussian=gaussian)
        #     self.entropy_net_CbCr = fractalar_large_in16(rle=rle, gaussian=gaussian)
        #     y_cond_emb_dim = 1024
        #     cbcr_cond_emb_dim = 512
        
        # elif net == 'H':
        #     self.entropy_net_Y = fractalar_huge_in32(rle=rle, gaussian=gaussian)
        #     self.entropy_net_CbCr = fractalar_huge_in16(rle=rle, gaussian=gaussian)
        #     y_cond_emb_dim = 1280
        #     cbcr_cond_emb_dim = 640

        if gaussian:
            self.gaussian_latent_encode = GaussianConditionalLatentCodec_ST_Trans
            print('using gaussian latent code')
        else:
            self.gaussian_latent_encode = PolynomialLaplaceConditionalLatentCodec_ST
            print('using laplace latent code')

        if rle:
            h_e_Y = nn.Sequential(
                conv(2*64*4, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )
        else:
            h_e_Y = nn.Sequential(
                conv(64*4, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )

        h_d_Y = nn.Sequential(
            conv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M, N, stride=1, kernel_size=3),
        )

        # if rle:
        #     h_e_Y = nn.Sequential(
        #         conv(2*64*4, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
                
        #     )
        # else:
        #     h_e_Y = nn.Sequential(
        #         conv(64*4, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #     )

        # h_d_Y = nn.Sequential(
        #     conv(N, N, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(N, M, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(M, M, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(M, N, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(N, N, stride=1, kernel_size=3),
        # )

        self.Y_cond_emb = nn.Linear(lsize*lsize*N, y_cond_emb_dim, bias=True)

        self.hyper_Y= HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_e_Y,
                    h_s=h_d_Y,
                    quantizer="ste",
                )

        if rle:
            h_e_C = nn.Sequential(
                conv(2*64, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )
        else:
            h_e_C = nn.Sequential(
                conv(64, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )

        h_d_C = nn.Sequential(
            conv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M, N, stride=1, kernel_size=3),
        )
        
        # if rle:
        #     h_e_C = nn.Sequential(
        #         conv(2*64, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #     )
        # else:
        #     h_e_C = nn.Sequential(
        #         conv(64, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=2, kernel_size=3),
        #         nn.SiLU(inplace=True),
        #         conv(N, N, stride=1, kernel_size=3),
        #     )

        # h_d_C = nn.Sequential(
        #     conv(N, N, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(N, M, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(M, M, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(M, N, stride=1, kernel_size=3),
        #     nn.SiLU(inplace=True),
        #     conv(N, N, stride=1, kernel_size=3),
        # )

        
        self.CbCr_cond_emb = nn.Linear(lsize*lsize*N, cbcr_cond_emb_dim, bias=True)

        self.hyper_cbcr= HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_e_C,
                    h_s=h_d_C,
                    quantizer="ste",
                )
        
        self.Gaussian_Y = self.gaussian_latent_encode(chunks=self.chunk,  entropy_parameters=self.entropy_net_Y)
        self.Gaussian_CbCr = self.gaussian_latent_encode(chunks=self.chunk,  entropy_parameters=self.entropy_net_CbCr)

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
    
    # def batch_rle_encode_full_expand(self, x: torch.Tensor, max_segments=3):
    #     B, N = x.shape
    #     device = x.device
    #     out = torch.zeros(B, N, dtype=x.dtype, device=device)
    #     mask = torch.zeros_like(out, dtype=torch.bool)

    #     for i in range(B):
    #         seq = x[i]
    #         if seq.numel() == 0:
    #             continue

    #         diffs = seq[1:] != seq[:-1]
    #         idx = torch.cat([
    #             torch.tensor([0], device=device),
    #             torch.nonzero(diffs).squeeze(-1) + 1,
    #             torch.tensor([seq.numel()], device=device)
    #         ])
    #         lengths = idx[1:] - idx[:-1]
    #         values = seq[idx[:-1]]

    #         # 前 max_segments 部分
    #         segs = min(max_segments, values.size(0))
    #         front = torch.stack([values[:segs], lengths[:segs]], dim=1).flatten()

    #         # 剩余部分
    #         if segs < values.size(0):
    #             rest = seq[lengths[:segs].sum():]
    #             full = torch.cat([front, rest], dim=0)
    #         else:
    #             full = front

    #         L = min(full.numel(), N)
    #         out[i, :L] = full[:L]
    #         mask[i, :L] = 1

    #     return out, mask

    def forward(self, Y, Cb, Cr): 
        bpp_likelihoods_z = 0
        bpp_likelihoods_y = 0
        bpp_likelihoods_cbcr = 0
        regs = 0
        CbCr = torch.cat((Cb, Cr), dim=0) # parallel 
        if self.rle:
            b, h, w, _ = Y.shape
            Y = Y.reshape(-1, 64)
            Y_ori = Y.clone()
            Y_ori = Y_ori.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            Y, Ymasks = rle_cuda.rle_encode(Y.clone().long(), 3)
            # Y_n, Ymasks_n = self.batch_rle_encode_full_expand(Y.clone().long(), max_segments=3)
            Y = Y.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            Ymasks = Ymasks.reshape(b, h, w, 64).permute(0, 3, 1, 2)

            b, h, w, _ = CbCr.shape
            CbCr = CbCr.reshape(-1, 64)
            CbCr_ori = CbCr.clone()
            CbCr_ori = CbCr_ori.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            CbCr, CbCrmasks = rle_cuda.rle_encode(CbCr.clone().long(), 3)
            # CbCr_n, CbCrmasks_n = self.batch_rle_encode_full_expand(CbCr.clone().long(), max_segments=3)
            CbCr = CbCr.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            CbCrmasks = CbCrmasks.reshape(b, h, w, 64).permute(0, 3, 1, 2)
            Y1, Y2, Y3, Y4 = self.split_Y(Y) 
            Y1_ori, Y2_ori, Y3_ori, Y4_ori = self.split_Y(Y_ori) 
            y_out = self.hyper_Y(torch.cat((Y1, Y2, Y3, Y4, Y1_ori, Y2_ori, Y3_ori, Y4_ori), dim=1))
        else:
            Y = Y.permute(0, 3, 1, 2) 
            CbCr = CbCr.permute(0, 3, 1, 2)
            
            Y1, Y2, Y3, Y4 = self.split_Y(Y) 
            y_out = self.hyper_Y(torch.cat((Y1, Y2, Y3, Y4), dim=1))
        z_y_likelihoods = y_out["likelihoods"]["z"]
        bpp_likelihoods_z += self.bpp_loss(z_y_likelihoods)  # z
        h_y = y_out["params"] 
        cond_Y = self.Y_cond_emb(h_y.reshape(-1, self.lsize*self.lsize*self.N))
        if self.rle:
            y_out = self.Gaussian_Y(Y, cond_Y, Ymasks)
        else:
            y_out = self.Gaussian_Y(Y, cond_Y, None)
            
        bpp_likelihoods_y += self.bpp_loss(y_out["likelihoods"]["y"]) # y

        # if self.rle:
        #     cbcr_out = self.hyper_cbcr(torch.cat((CbCr, CbCr_ori), dim=1)) 
        # else:
        #     cbcr_out = self.hyper_cbcr(CbCr)
        # z_cbcr_likelihoods = cbcr_out["likelihoods"]["z"]
        # bpp_likelihoods_z += self.bpp_loss(z_cbcr_likelihoods)  # z
        # h_cbcr = cbcr_out["params"] 
        # cond_CbCr = self.CbCr_cond_emb(h_cbcr.reshape(-1, self.lsize*self.lsize*self.N))

        # if self.rle:
        #     CbCr_out = self.Gaussian_CbCr(CbCr, cond_CbCr, CbCrmasks)
        # else:
        #     CbCr_out = self.Gaussian_CbCr(CbCr, cond_CbCr, None)
        # bpp_likelihoods_cbcr += self.bpp_loss(CbCr_out["likelihoods"]["y"]) # cbcr
        # regs += CbCr_out["reg"]
        # regs += y_out["reg"]
        return {"bpp_loss": bpp_likelihoods_z + bpp_likelihoods_y + bpp_likelihoods_cbcr, "reg": regs}
        
       



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

#### base  444

# def fractalar_base_in32(rle, gaussian, **kwargs):  #small
#     if gaussian:
#         channel = 2
#     else:
#         channel = 3
#     model = FractalGen(
#         img_size_list=(32, 16, 4, 1),
#             embed_dim_list=(768, 384, 192, 64),
#             num_blocks_list=(6, 6, 3, 1), #1
#             num_heads_list=(6, 6, 3, 4),
#             generator_type_list=("ar", "ar", "ar", "ar"),
#             fractal_level=0,
#             rle=rle,
#             channel=channel,
#         **kwargs)
#     return model


# def fractalar_base_in16(rle, gaussian, **kwargs):
#     if gaussian:
#         channel = 2
#     else:
#         channel = 3
#     model = FractalGen(
#         img_size_list=(16, 4, 1),
#             embed_dim_list=(384, 192, 64), ##(384, 192, 64)
#             num_blocks_list=(6, 3, 1), ##(6, 3, 1)
#             num_heads_list=(6, 3, 4), ##(6, 3, 4)
#             generator_type_list=("ar", "ar", "ar"),
#             fractal_level=0,
#             rle=rle,
#             channel=channel,
#         **kwargs)
#     return model



try:
    import rle_cuda
except ImportError:
    print("CUDA extension not found. Compiling from source...")

class TransJPEGRecompression444(CompressionModel):
    """Efficient JPEG recompression model.
    Args:
        N (int): Number of channels
        chunk (tuple): chunk type  such as ("scales", "means")
    """


    def __init__(self, N=192, M=288, chunk=("scales", "means"), lsize=4, net='B', rle=False, gaussian=False, **kwargs): # lsize 4 (256)  12 (768)
        super().__init__(**kwargs)
        self.N = N
        self.M = M
        self.chunk = chunk
        self.lsize = lsize
        self.rle = rle
        self.gaussian = gaussian
        if net == 'B':
            self.entropy_net_Y = fractalar_base_in32(rle=rle, gaussian=gaussian)
            self.entropy_net_CbCr = fractalar_base_in32(rle=rle, gaussian=gaussian)
            y_cond_emb_dim = 768
            cbcr_cond_emb_dim = 768
        # elif net == 'L':
        #     self.entropy_net_Y = fractalar_large_in32(rle=rle, gaussian=gaussian)
        #     self.entropy_net_CbCr = fractalar_large_in16(rle=rle, gaussian=gaussian)
        #     y_cond_emb_dim = 1024
        #     cbcr_cond_emb_dim = 512
        
        # elif net == 'H':
        #     self.entropy_net_Y = fractalar_huge_in32(rle=rle, gaussian=gaussian)
        #     self.entropy_net_CbCr = fractalar_huge_in16(rle=rle, gaussian=gaussian)
        #     y_cond_emb_dim = 1280
        #     cbcr_cond_emb_dim = 640

        if gaussian:
            self.gaussian_latent_encode = GaussianConditionalLatentCodec_ST_Trans
            print('using gaussian latent code')
        else:
            self.gaussian_latent_encode = PolynomialLaplaceConditionalLatentCodec_ST
            print('using laplace latent code')

        if rle:
            h_e_Y = nn.Sequential(
                conv(2*64*4, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )
        else:
            h_e_Y = nn.Sequential(
                conv(64*4, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )

        h_d_Y = nn.Sequential(
            conv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M, N, stride=1, kernel_size=3),
        )



        self.Y_cond_emb = nn.Linear(lsize*lsize*N, y_cond_emb_dim, bias=True)

        self.hyper_Y= HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_e_Y,
                    h_s=h_d_Y,
                    quantizer="ste",
                )

        if rle:
            h_e_C = nn.Sequential(
                conv(2*64, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )
        else:
            h_e_C = nn.Sequential(
                conv(64*4, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=3),
            )

        h_d_C = nn.Sequential(
            conv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M, N, stride=1, kernel_size=3),
        )
        
        
        
        self.CbCr_cond_emb = nn.Linear(lsize*lsize*N, cbcr_cond_emb_dim, bias=True)

        self.hyper_cbcr= HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_e_C,
                    h_s=h_d_C,
                    quantizer="ste",
                )
        
        self.Gaussian_Y = self.gaussian_latent_encode(chunks=self.chunk,  entropy_parameters=self.entropy_net_Y)
        self.Gaussian_CbCr = self.gaussian_latent_encode(chunks=self.chunk,  entropy_parameters=self.entropy_net_CbCr)

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
    
    # def batch_rle_encode_full_expand(self, x: torch.Tensor, max_segments=3):
    #     B, N = x.shape
    #     device = x.device
    #     out = torch.zeros(B, N, dtype=x.dtype, device=device)
    #     mask = torch.zeros_like(out, dtype=torch.bool)

    #     for i in range(B):
    #         seq = x[i]
    #         if seq.numel() == 0:
    #             continue

    #         diffs = seq[1:] != seq[:-1]
    #         idx = torch.cat([
    #             torch.tensor([0], device=device),
    #             torch.nonzero(diffs).squeeze(-1) + 1,
    #             torch.tensor([seq.numel()], device=device)
    #         ])
    #         lengths = idx[1:] - idx[:-1]
    #         values = seq[idx[:-1]]

    #         # 前 max_segments 部分
    #         segs = min(max_segments, values.size(0))
    #         front = torch.stack([values[:segs], lengths[:segs]], dim=1).flatten()

    #         # 剩余部分
    #         if segs < values.size(0):
    #             rest = seq[lengths[:segs].sum():]
    #             full = torch.cat([front, rest], dim=0)
    #         else:
    #             full = front

    #         L = min(full.numel(), N)
    #         out[i, :L] = full[:L]
    #         mask[i, :L] = 1

    #     return out, mask

    def forward(self, Y, Cb, Cr): 
        bpp_likelihoods_z = 0
        bpp_likelihoods_y = 0
        bpp_likelihoods_cbcr = 0
        regs = 0
        CbCr = torch.cat((Cb, Cr), dim=0) # parallel 
        if self.rle:
            b, h, w, _ = Y.shape
            Y = Y.reshape(-1, 64)
            Y_ori = Y.clone()
            Y_ori = Y_ori.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            Y, Ymasks = rle_cuda.rle_encode(Y.clone().long(), 3)
            # Y_n, Ymasks_n = self.batch_rle_encode_full_expand(Y.clone().long(), max_segments=3)
            Y = Y.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            Ymasks = Ymasks.reshape(b, h, w, 64).permute(0, 3, 1, 2)

            b, h, w, _ = CbCr.shape
            CbCr = CbCr.reshape(-1, 64)
            CbCr_ori = CbCr.clone()
            CbCr_ori = CbCr_ori.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            CbCr, CbCrmasks = rle_cuda.rle_encode(CbCr.clone().long(), 3)
            # CbCr_n, CbCrmasks_n = self.batch_rle_encode_full_expand(CbCr.clone().long(), max_segments=3)
            CbCr = CbCr.reshape(b, h, w, 64).permute(0, 3, 1, 2).float()
            CbCrmasks = CbCrmasks.reshape(b, h, w, 64).permute(0, 3, 1, 2)
            Y1, Y2, Y3, Y4 = self.split_Y(Y) 
            Y1_ori, Y2_ori, Y3_ori, Y4_ori = self.split_Y(Y_ori) 
            y_out = self.hyper_Y(torch.cat((Y1, Y2, Y3, Y4, Y1_ori, Y2_ori, Y3_ori, Y4_ori), dim=1))
        else:
            Y = Y.permute(0, 3, 1, 2) 
            CbCr = CbCr.permute(0, 3, 1, 2)
            
            Y1, Y2, Y3, Y4 = self.split_Y(Y) 
            y_out = self.hyper_Y(torch.cat((Y1, Y2, Y3, Y4), dim=1))
        z_y_likelihoods = y_out["likelihoods"]["z"]
        bpp_likelihoods_z += self.bpp_loss(z_y_likelihoods)  # z
        h_y = y_out["params"] 
        cond_Y = self.Y_cond_emb(h_y.reshape(-1, self.lsize*self.lsize*self.N))
        if self.rle:
            y_out = self.Gaussian_Y(Y, cond_Y, Ymasks)
        else:
            y_out = self.Gaussian_Y(Y, cond_Y, None)
            
        bpp_likelihoods_y += self.bpp_loss(y_out["likelihoods"]["y"]) # y

        if self.rle:
            cbcr_out = self.hyper_cbcr(torch.cat((CbCr, CbCr_ori), dim=1)) 
        else:
            CbCr1, CbCr2, CbCr3, CbCr4 = self.split_Y(CbCr)
            cbcr_out = self.hyper_cbcr(torch.cat((CbCr1, CbCr2, CbCr3, CbCr4), dim=1))
        z_cbcr_likelihoods = cbcr_out["likelihoods"]["z"]
        bpp_likelihoods_z += self.bpp_loss(z_cbcr_likelihoods)  # z
        h_cbcr = cbcr_out["params"] 
        cond_CbCr = self.CbCr_cond_emb(h_cbcr.reshape(-1, self.lsize*self.lsize*self.N))

        if self.rle:
            CbCr_out = self.Gaussian_CbCr(CbCr, cond_CbCr, CbCrmasks)
        else:
            CbCr_out = self.Gaussian_CbCr(CbCr, cond_CbCr, None)
        bpp_likelihoods_cbcr += self.bpp_loss(CbCr_out["likelihoods"]["y"]) # y
        regs += CbCr_out["reg"]
        regs += y_out["reg"]
        return {"bpp_loss": bpp_likelihoods_z + bpp_likelihoods_y + bpp_likelihoods_cbcr, "reg": regs}
        
        # return {"bpp_loss": bpp_likelihoods_z + bpp_likelihoods_y + bpp_likelihoods_cbcr}



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