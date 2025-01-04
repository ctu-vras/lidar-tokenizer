import torch
import torch.nn as nn
from functools import partial

try:
    from model_managment.helpers import ResnetBlock, ResnetBlock2, make_attn, Downsample, Downsample2, Normalize, nonlinearity, CircularConv2d,get_2d_sincos_pos_embed
except Exception as e:
    print(e)
    print("\nTRYING WITH RELATIVE IMPORTS, BUT IF THE ISSUE WAS SOMETHING ELSE THEN YOU ARE PROBABLY MISSING A PACKAGE\n")
    from ..model_managment.helpers import ResnetBlock, ResnetBlock2, make_attn, Downsample, Downsample2, Normalize, nonlinearity, CircularConv2d,get_2d_sincos_pos_embed

from timm.models.vision_transformer import PatchEmbed
from timm.models.swin_transformer import BasicLayer

""" From https://github.com/myc634/UltraLiDAR_nusc_waymo """
class VQEncoder(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size=8,
        in_chans=53,
        embed_dim=512,
        num_heads=16,
        depth=12,
        codebook_dim=1024,
    ):
        super().__init__()

        norm_layer = nn.LayerNorm
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, norm_layer=norm_layer)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        if isinstance(img_size, tuple):
            self.h = img_size[0] // patch_size
            self.w = img_size[1] // patch_size
        else:
            self.h = img_size // patch_size
            self.w = img_size // patch_size

        self.blocks = [
            BasicLayer(
                embed_dim,
                (self.h, self.w),
                depth,
                num_heads=num_heads,
                window_size=8,
                downsample=None,
                # use_checkpoint=False,
            ),
        ]

        self.blocks = nn.Sequential(*self.blocks)

        self.norm = nn.Sequential(norm_layer(embed_dim), nn.GELU())
        self.pre_quant = nn.Linear(embed_dim, codebook_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # nn.init.constant_(self.pre_quant.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pre_quant(x)

        return x

    
""" From https://github.com/hancyran/LiDAR-Diffusion """
class LidarDiffusionEncoder(nn.Module):
    """ This one doesn't have the curve-wise blocks!! """
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_levels, dropout=0.0, resamp_with_conv=True, in_channels,
                 z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if i_level in attn_levels:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

""" From https://github.com/hancyran/LiDAR-Diffusion """ 
class LidarDiffusionEncoder2(nn.Module):
    """ This one has the curve-wise blocks. It is the one they talk about in the LD paper. """
    def __init__(self, *, ch, out_ch, ch_mult, strides, num_res_blocks,
                 attn_levels, dropout=0.0, resamp_with_conv=True, in_channels, z_channels,
                 double_z=True, use_linear_attn=False, attn_type="vanilla", use_mask=False,
                 **ignore_kwargs):
        super().__init__()
        if use_mask:
            assert out_ch == in_channels + 1, 'Set "out_ch = out_ch + 1" for mask prediction.'
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # downsampling
        self.conv_in = CircularConv2d(in_channels,
                                      self.ch,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock2(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if i_level in attn_levels:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                stride = tuple(strides[i_level])
                down.downsample = Downsample2(block_in, resamp_with_conv, stride)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock2(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = CircularConv2d(block_in,
                                       2 * z_channels if double_z else z_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h