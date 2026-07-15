from functools import partial

import torch
import torch.nn as nn

from compressai.models.ar import AR, AR_Last

class FractalGen(nn.Module):
    """ Fractal Generative Model"""

    def __init__(self,
                 img_size_list,
                 embed_dim_list,
                 num_blocks_list,
                 num_heads_list,
                 generator_type_list,
                 label_drop_prob=0.1,
                 
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 guiding_pixel=False,
                 num_conds=1,
                 r_weight=1.0,
                 grad_checkpointing=False,
                 fractal_level=0,
                 rle=False,
                 channel=2,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # fractal specifics
        self.fractal_level = fractal_level
        self.num_fractal_levels = len(img_size_list)
        self.embed_dim_list = embed_dim_list
        self.rle = rle
        self.channel = channel


        # --------------------------------------------------------------------------
        # Generator for the current level
        if generator_type_list[fractal_level] == "ar":
            generator = AR
        else:
            raise NotImplementedError
        self.generator = generator(
            seq_len=(img_size_list[fractal_level] // img_size_list[fractal_level+1]) ** 2,
            patch_size=img_size_list[fractal_level+1],
            cond_embed_dim=embed_dim_list[fractal_level-1] if fractal_level > 0 else embed_dim_list[0],
            embed_dim=embed_dim_list[fractal_level],
            num_blocks=num_blocks_list[fractal_level],
            num_heads=num_heads_list[fractal_level],
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            guiding_pixel=guiding_pixel if fractal_level > 0 else False,
            num_conds=num_conds,
            grad_checkpointing=grad_checkpointing,

        )

        # --------------------------------------------------------------------------
        # Build the next fractal level recursively
        if self.fractal_level < self.num_fractal_levels - 2:
            self.next_fractal = FractalGen(
                img_size_list=img_size_list,
                embed_dim_list=embed_dim_list,
                num_blocks_list=num_blocks_list,
                num_heads_list=num_heads_list,
                generator_type_list=generator_type_list,
                label_drop_prob=label_drop_prob,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                guiding_pixel=guiding_pixel,
                num_conds=num_conds,
                r_weight=r_weight,
                grad_checkpointing=grad_checkpointing,
                fractal_level=fractal_level+1,
                rle=self.rle
            )
        else:
            # The final fractal level uses PixelLoss.
            self.next_fractal = AR_Last(
            seq_len=embed_dim_list[fractal_level+1],
            cond_embed_dim=embed_dim_list[fractal_level],
            embed_dim=embed_dim_list[fractal_level+1],
            num_blocks=num_blocks_list[fractal_level+1],
            num_heads=num_heads_list[fractal_level+1],
            rle=self.rle,
            channel=self.channel
            )

    def forward(self, imgs, cond_list = None, masks=None):
        """
        Forward pass to get loss recursively.
        """
        # Get image patches and conditions for the next level
        # print('imgs', imgs.shape)
        imgs, cond_list, masks = self.generator(imgs, cond_list, masks)  #[1, 3, 256, 256] [1, 768] [256, 3, 16, 16] [256, 768] [4096, 3, 4, 4] [4096, 384]
        # Compute loss recursively from the next fractal level.
        loss = self.next_fractal(imgs, cond_list, masks)
        return loss
    
    def sample(self, cond_list, num_iter_list, fractal_level):
        """
        Generate samples recursively.
        """
        if fractal_level < self.num_fractal_levels - 2:
            next_level_sample_function = partial(
                self.next_fractal.sample,
                num_iter_list=num_iter_list,
                fractal_level=fractal_level + 1
            )
        else:
            next_level_sample_function = self.next_fractal.sample

        # Recursively sample using the current generator.
        return self.generator.sample(
            cond_list, num_iter_list[fractal_level],next_level_sample_function)