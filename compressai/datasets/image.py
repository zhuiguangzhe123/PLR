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

from pathlib import Path
try:
    import jpegio
except:
    pass
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F

from compressai.registry import register_dataset
import os
import cv2
import numpy as np
import torch
import random


@register_dataset("ImageFolder")
class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        # self.samples = sorted(f for f in splitdir.iterdir() if f.is_file())
        self.samples = []
        for i in os.listdir(splitdir):
            self.samples += [os.path.join(splitdir, i, j) for j in os.listdir(os.path.join(splitdir, i))]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        # img = Image.open(self.samples[index]).convert("RGB")
        img = cv2.imread(self.samples[index])
        img = self.normalize_(img).transpose(2, 0, 1)
        return img

    def normalize_(self, img):
        img = img.astype(np.float32)
        img /= 255.0
        return img
        

    def __len__(self):
        return len(self.samples)







def generate_zigzag_indices(n=8):
    """ 生成n x n矩阵的Zig-zag索引顺序 """
    indices = []
    for i in range(2*n-1):
        if i % 2 == 0:  # 偶数对角线：从下往上
            x = min(i, n-1)
            y = max(0, i - n + 1)
            while x >= 0 and y < n:
                indices.append((x, y) if i < n else (y, x))
                x -= 1
                y += 1
        else:           # 奇数对角线：从上往下
            x = max(0, i - n + 1)
            y = min(i, n-1)
            while x < n and y >= 0:
                indices.append((x, y) if i < n else (y, x))
                x += 1
                y -= 1
    indices = torch.tensor(indices)
    indices = indices[:, 0] * n + indices[:, 1]  # 转换为行索引
    return indices


def dct_processing(input_tensor, zig_indices, n=8):
    """
    输入:NxN的DCT系数矩阵
    输出:经过Zig-zag扫描和逆序处理的[N//8,N//8,64]矩阵
    """
    # 步骤1:分割为8x8块
    H, W = input_tensor.shape
    blocks = input_tensor.reshape(H//n, n, W//n, n).permute(0, 2, 1, 3)
    # 步骤2:Zig-zag扫描
    scanned_blocks = blocks.reshape(H//n, W//n, n*n)[..., zig_indices]
    # 步骤3:逆序处理每个块内的元素
    reversed_blocks = scanned_blocks.flip(dims=[-1])
    
    return reversed_blocks


def dct_processing_inverse(reversed_scanned_blocks, zig_indices, n=8):
    b1, b2, _ = reversed_scanned_blocks.shape
    reversed_scanned_blocks = reversed_scanned_blocks.flip(dims=[-1])  # 逆序处理每个块内的元素
    blocks = torch.zeros((b1, b2, n * n), dtype=reversed_scanned_blocks.dtype)
    # 按照 Zig-zag 索引填充回原来的位置
    blocks.scatter_(dim=2, index=zig_indices.expand(b1, b2, -1), src=reversed_scanned_blocks)
    return blocks.view(b1, b2, n, n).permute(0, 2, 1, 3).reshape(b1*n, b2*n)


@register_dataset("DCTFolder")
class DCTFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, txt_root, transform=None, split="train"):
        splitdir = Path(root) / split
        txt_420_path = os.path.join(txt_root, split+'_paths_420.txt')
        txt_422_path = os.path.join(txt_root, split+'_paths_422.txt')

        with open(txt_422_path, "r") as f_in:
            self.paths_422 = [f"{splitdir}/{f.strip()}" for f in f_in]

        with open(txt_420_path, "r") as f_in:
            self.paths_420 = [f"{splitdir}/{f.strip()}" for f in f_in]

        # if not splitdir.is_dir():
        #     raise RuntimeError(f'Missing directory "{splitdir}"')

        # self.samples = []
        # for i in os.listdir(splitdir):
        #     self.samples += [os.path.join(splitdir, i, j) for j in os.listdir(os.path.join(splitdir, i))]

        if len(self.paths_420) > len(self.paths_422):
            scale = (len(self.paths_420) // len(self.paths_422))+1
            self.paths_422 = self.paths_422 * scale
            self.paths_422 = self.paths_422[:len(self.paths_420)]
        elif len(self.paths_422) > len(self.paths_420):
            scale = (len(self.paths_422) // len(self.paths_420))+1
            self.paths_420 = self.paths_420 * scale
            self.paths_420 = self.paths_420[:len(self.paths_422)]
        self.transform = transform
        self.zig_indice = generate_zigzag_indices(n=8)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        jpeg_struct = jpegio.read(self.paths_422[index])
        Y_zigzag = dct_processing(torch.from_numpy(jpeg_struct.coef_arrays[0]), self.zig_indice, n=8).float()
        Cb = torch.from_numpy(jpeg_struct.coef_arrays[1]).float()
        Cr = torch.from_numpy(jpeg_struct.coef_arrays[2]).float()
        # if Cb.shape[0] == 128:
        #     Cb = F.pad(Cb, (0, 0, 0, 256-Cb.shape[0]), 'constant', value=0)
        #     Cr = F.pad(Cr, (0, 0, 0, 256-Cr.shape[0]), 'constant', value=0)
        jpeg_struct = jpegio.read(self.paths_420[index])
        Y_zigzag_420 = dct_processing(torch.from_numpy(jpeg_struct.coef_arrays[0]), self.zig_indice, n=8).float()
        Cb_420 = torch.from_numpy(jpeg_struct.coef_arrays[1]).float()
        Cr_420 = torch.from_numpy(jpeg_struct.coef_arrays[2]).float()
        return {'Y': Y_zigzag, 'Cb': Cb.unsqueeze(0), 'Cr': Cr.unsqueeze(0)}, {'Y': Y_zigzag_420, 'Cb': Cb_420.unsqueeze(0), 'Cr': Cr_420.unsqueeze(0)}
        

    def __len__(self):
        return len(self.paths_420)





import torchjpeg.codec as tc
from torchvision.transforms.functional import to_pil_image, to_tensor
class PNGFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, mode="train", quality=75):
        dirs = os.listdir(root)
        if mode == 'train':
            dirs = [d for d in dirs if int(d.split('-')[0].split('_')[-1])<=100]
        else:
            dirs = [d for d in dirs if int(d.split('-')[0].split('_')[-1])>100]
        self.paths = []
        for d in dirs:
            self.paths += [os.path.join(root, d, f) for f in os.listdir(os.path.join(root, d))]
        
        self.quality = quality

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        image = Image.open(self.paths[index]) # h, w, 3
        image = to_tensor(image)
        size, quality_table, Y, CbCr  = tc.quantize_at_quality(image, self.quality, 1, 1)
        yh, yw = Y.shape[1], Y.shape[2]
        Y_zigzag = Y[0].reshape(yh, yw, -1) # 低频向高频排列
        Y_zigzag = torch.flip(Y_zigzag, dims=[-1]) # 高频向低频排列
        ch, cw = CbCr.shape[1], CbCr.shape[2]
        Cb = CbCr[0]
        Cb = Cb.permute(0, 2, 1, 3).reshape(ch*8, cw*8)
        Cr = CbCr[1]
        Cr = Cr.permute(0, 2, 1, 3).reshape(ch*8, cw*8)

        return {'Y': Y_zigzag.float(), 'Cb': Cb.unsqueeze(0).float(), 'Cr': Cr.unsqueeze(0).float()}
        # ys = []
        # Cbs = []
        # Crs = []
        # image = Image.open(self.paths[index]) # h, w, 3
        # for i in range(3):
        #     for j in range(3):
        #         image_new = image.crop((i*256, j*256, (i+1)*256, (j+1)*256))
        #         image_new = to_tensor(image_new)
        #         size, quality_table, Y, CbCr  = tc.quantize_at_quality(image_new, self.quality, 1, 1)
        #         yh, yw = Y.shape[1], Y.shape[2]
        #         Y_zigzag = Y[0].reshape(yh, yw, -1) # 低频向高频排列
        #         Y_zigzag = torch.flip(Y_zigzag, dims=[-1]) # 高频向低频排列
        #         # a, b = batch_rle_encode_full_expand(Y_zigzag[0, :2, :])
        #         ch, cw = CbCr.shape[1], CbCr.shape[2]
        #         Cb = CbCr[0]
        #         Cb = Cb.permute(0, 2, 1, 3).reshape(ch*8, cw*8)
        #         Cr = CbCr[1]
        #         Cr = Cr.permute(0, 2, 1, 3).reshape(ch*8, cw*8)
        #         ys.append(Y_zigzag.float())
        #         Cbs.append(Cb.unsqueeze(0).float())
        #         Crs.append(Cr.unsqueeze(0).float())


        # return {'Y': torch.stack(ys), 'Cb': torch.stack(Cbs), 'Cr': torch.stack(Crs)}
        

    def __len__(self):
        return len(self.paths)
    
class PNGFolder_TCGA_test(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, mode="train", quality=75):
        dirs = os.listdir(root)
        self.paths = []
        for d in dirs:
            self.paths += [os.path.join(root, d, f) for f in os.listdir(os.path.join(root, d))]
        
        self.quality = quality

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        image = Image.open(self.paths[index]) # h, w, 3
        image = to_tensor(image)
        size, quality_table, Y, CbCr  = tc.quantize_at_quality(image, self.quality, 1, 1)
        yh, yw = Y.shape[1], Y.shape[2]
        Y_zigzag = Y[0].reshape(yh, yw, -1) # 低频向高频排列
        Y_zigzag = torch.flip(Y_zigzag, dims=[-1]) # 高频向低频排列
        ch, cw = CbCr.shape[1], CbCr.shape[2]
        Cb = CbCr[0]
        Cb = Cb.permute(0, 2, 1, 3).reshape(ch*8, cw*8)
        Cr = CbCr[1]
        Cr = Cr.permute(0, 2, 1, 3).reshape(ch*8, cw*8)

        return {'Y': Y_zigzag.float(), 'Cb': Cb.unsqueeze(0).float(), 'Cr': Cr.unsqueeze(0).float()}


    def __len__(self):
        return len(self.paths)
    
class PNGFolder_Trans_TCGA_test(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, mode="train", quality=75):
        dirs = os.listdir(root)
        self.paths = []
        for d in dirs:
            self.paths += [os.path.join(root, d, f) for f in os.listdir(os.path.join(root, d))]
        
        self.quality = quality
        self.mode = mode


    def normalize_(self, img):
        img = img.astype(np.float32)
        img /= 255.0
        return img

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        # # image = Image.open(self.paths[index]) # h, w, 3
        image = cv2.imread(self.paths[index])
        image = torch.from_numpy(self.normalize_(image).transpose(2, 0, 1)) #[c, h, w]
        # image = to_tensor(image)
        size, quality_table, Y, CbCr  = tc.quantize_at_quality(image, self.quality, 2, 2)  #default 2, 2
        yh, yw = Y.shape[1], Y.shape[2]
        Y_zigzag = Y[0].reshape(yh, yw, -1) # 低频向高频排列
        Y_zigzag = torch.flip(Y_zigzag, dims=[-1]) # 高频向低频排列

        #order test
        # spacial = (np.array(list(range(0, 1024)))+1).reshape(32,32,1)
        # spacial = torch.from_numpy(spacial).float()
        # spacial = spacial.repeat(1,1,64)
        # Y_zigzag = spacial
        ######

        ch, cw = CbCr.shape[1], CbCr.shape[2]
        Cb = CbCr[0].reshape(ch, cw, -1)
        Cb = torch.flip(Cb, dims=[-1]) # 高频向低频排列
        Cr = CbCr[1].reshape(ch, cw, -1)
        Cr = torch.flip(Cr, dims=[-1]) # 高频向低频排列
        return {'Y': Y_zigzag.float(), 'Cb': Cb.float(), 'Cr': Cr.float()}
    
    def __len__(self):
        return len(self.paths)
    

class PNGFolder_Trans(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, mode="train", quality=75):
        dirs = os.listdir(root)
        if mode == 'train':
            dirs = [d for d in dirs if int(d.split('-')[0].split('_')[-1])<=100]
        elif mode == 'test':
            dirs = [d for d in dirs if int(d.split('-')[0].split('_')[-1])>100]
        else:
            dirs = [d for d in dirs if int(d.split('-')[0].split('_')[-1])>128]
        self.paths = []
        for d in dirs:
            self.paths += [os.path.join(root, d, f) for f in os.listdir(os.path.join(root, d))]
        
        self.quality = quality
        self.mode = mode


    def normalize_(self, img):
        img = img.astype(np.float32)
        img /= 255.0
        return img

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        # # image = Image.open(self.paths[index]) # h, w, 3
        image = cv2.imread(self.paths[index])
        image = torch.from_numpy(self.normalize_(image).transpose(2, 0, 1)) #[c, h, w]
        # image = to_tensor(image)
        size, quality_table, Y, CbCr  = tc.quantize_at_quality(image, self.quality, 2, 2)  #default 2, 2
        yh, yw = Y.shape[1], Y.shape[2]
        Y_zigzag = Y[0].reshape(yh, yw, -1) # 低频向高频排列
        Y_zigzag = torch.flip(Y_zigzag, dims=[-1]) # 高频向低频排列
        ch, cw = CbCr.shape[1], CbCr.shape[2]
        Cb = CbCr[0].reshape(ch, cw, -1)
        Cb = torch.flip(Cb, dims=[-1]) # 高频向低频排列
        Cr = CbCr[1].reshape(ch, cw, -1)
        Cr = torch.flip(Cr, dims=[-1]) # 高频向低频排列
        return {'Y': Y_zigzag.float(), 'Cb': Cb.float(), 'Cr': Cr.float()}
        # ys = []
        # Cbs = []
        # Crs = []
        # if self.mode == 'train':
        #     indices = torch.randperm(9)[:2]
        #     count = -1
        #     for i in range(3):
        #         for j in range(3):
        #             count += 1
        #             if count in indices:
        #                 image_new = image.crop((i*256, j*256, (i+1)*256, (j+1)*256))
        #                 image_new = to_tensor(image_new)
        #                 size, quality_table, Y, CbCr  = tc.quantize_at_quality(image_new, self.quality, 2, 2)
        #                 yh, yw = Y.shape[1], Y.shape[2]
        #                 Y_zigzag = Y[0].reshape(yh, yw, -1) # 低频向高频排列
        #                 Y_zigzag = torch.flip(Y_zigzag, dims=[-1]) # 高频向低频排列
        #                 ch, cw = CbCr.shape[1], CbCr.shape[2]
        #                 Cb = CbCr[0].reshape(ch, cw, -1)
        #                 Cb = torch.flip(Cb, dims=[-1]) # 高频向低频排列
        #                 Cr = CbCr[1].reshape(ch, cw, -1)
        #                 Cr = torch.flip(Cr, dims=[-1]) # 高频向低频排列
        #                 ys.append(Y_zigzag.float())
        #                 Cbs.append(Cb.float())
        #                 Crs.append(Cr.float())
        #     return {'Y': torch.stack(ys), 'Cb': torch.stack(Cbs), 'Cr': torch.stack(Crs)}
        # else:
        #     for i in range(3):
        #         for j in range(3):
        #             image_new = image.crop((i*256, j*256, (i+1)*256, (j+1)*256))
        #             image_new = to_tensor(image_new)
        #             size, quality_table, Y, CbCr  = tc.quantize_at_quality(image_new, self.quality, 2, 2)
        #             yh, yw = Y.shape[1], Y.shape[2]
        #             Y_zigzag = Y[0].reshape(yh, yw, -1) # 低频向高频排列
        #             Y_zigzag = torch.flip(Y_zigzag, dims=[-1]) # 高频向低频排列
        #             ch, cw = CbCr.shape[1], CbCr.shape[2]
        #             Cb = CbCr[0].reshape(ch, cw, -1)
        #             Cb = torch.flip(Cb, dims=[-1]) # 高频向低频排列
        #             Cr = CbCr[1].reshape(ch, cw, -1)
        #             Cr = torch.flip(Cr, dims=[-1]) # 高频向低频排列
        #             ys.append(Y_zigzag.float())
        #             Cbs.append(Cb.float())
        #             Crs.append(Cr.float())
        
        #     return {'Y': torch.stack(ys), 'Cb': torch.stack(Cbs), 'Cr': torch.stack(Crs)}
        

    def __len__(self):
        return len(self.paths)
    




class PNGFolder_Foundation(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, mode="train", quality=[100, 95, 85, 75, 65, 55], txt_path='train.txt'):  #root:TGCA
        if mode == 'train':
            with open(os.path.join(root, txt_path), 'r') as f:
                dirs = [d.strip() for d in f.readlines()]
        elif mode == 'test':
            with open(os.path.join(root, txt_path), 'r') as f:
                dirs = [d.strip() for d in f.readlines()]
        else:
            with open(os.path.join(root, txt_path), 'r') as f:
                dirs = [d.strip() for d in f.readlines()]

        # dirs = dirs[:3600000]
        self.paths = [os.path.join(root, d) for d in dirs]
        self.quality = quality
        self.mode = mode


    def normalize_(self, img):
        img = img.astype(np.float32)
        img /= 255.0
        return img

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        # if self.mode == 'train':
        #     index = random.randint(0, len(self.quality) - 1)
        # else:
        #     index = 3
        # quality = self.quality[index]
        quality = 75
        
        image = cv2.imread(self.paths[index])
        image = torch.from_numpy(self.normalize_(image).transpose(2, 0, 1)) #[c, h, w]

        size, quality_table, Y, CbCr  = tc.quantize_at_quality(image, quality, 2, 2)
        yh, yw = Y.shape[1], Y.shape[2]
        Y_zigzag = Y[0].reshape(yh, yw, -1) # 低频向高频排列
        Y_zigzag = torch.flip(Y_zigzag, dims=[-1]) # 高频向低频排列
        # a, b = batch_rle_encode_full_expand(Y_zigzag[0, :2, :])
        ch, cw = CbCr.shape[1], CbCr.shape[2]
        Cb = CbCr[0]
        Cb = Cb.permute(0, 2, 1, 3).reshape(ch*8, cw*8)
        Cr = CbCr[1]
        Cr = Cr.permute(0, 2, 1, 3).reshape(ch*8, cw*8)

        return {'Y': Y_zigzag.float(), 'Cb': Cb.unsqueeze(0).float(), 'Cr': Cr.unsqueeze(0).float()}
        

    def __len__(self):
        return len(self.paths)