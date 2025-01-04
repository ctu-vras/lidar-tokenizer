import torch
import torch.nn as nn
try:
    from model_managment.helpers import ResnetBlock, ResnetBlock2, make_attn, Normalize, nonlinearity, Upsample, Upsample2, CircularConv2d,get_2d_sincos_pos_embed
except Exception as e:
    print(e)
    print("\nTRYING WITH RELATIVE IMPORTS, BUT IF THE ISSUE WAS SOMETHING ELSE THEN YOU ARE PROBABLY MISSING A PACKAGE\n")
    from ..model_managment.helpers import ResnetBlock, ResnetBlock2, make_attn, Normalize, nonlinearity, Upsample, Upsample2, CircularConv2d,get_2d_sincos_pos_embed
from functools import partial

from timm.models.vision_transformer import PatchEmbed
from timm.models.swin_transformer import BasicLayer

""" From https://github.com/myc634/UltraLiDAR_nusc_waymo """
class VQDecoder(nn.Module):
    def __init__(
        self,
        img_size,
        num_patches,
        patch_size=8,
        in_chans=53,
        embed_dim=512,
        num_heads=16,
        depth=12,
        codebook_dim=1024,
        bias_init=-3,
    ):
        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        norm_layer = nn.LayerNorm
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size
        self.num_patches = num_patches
        self.decoder_embed = nn.Linear(codebook_dim, embed_dim, bias=True)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = BasicLayer(
            embed_dim,
            (img_size[0] // patch_size, img_size[1] // patch_size),
            depth=depth,
            num_heads=num_heads,
            window_size=8,
        )

        self.norm = nn.Sequential(norm_layer(embed_dim), nn.GELU())
        self.pred = nn.Linear(embed_dim, patch_size**2 * in_chans, bias=True)
        self.initialize_weights()
        nn.init.constant_(self.pred.bias, bias_init)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        p = self.patch_size
        # h = w = int(x.shape[1] ** 0.5)
        h, w = self.h, self.w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, w * p))

        return imgs

    def forward(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # predictor projection
        x = self.pred(x)
        x = self.unpatchify(x)

        return x



""" From https://github.com/hancyran/LiDAR-Diffusion """
class LidarDiffusionDecoder(nn.Module):
    """ This one doesn't have the curve-wise blocks!! """
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_levels, dropout=0.0, resamp_with_conv=True, in_channels,
                 z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

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

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if i_level in attn_levels:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
    
""" From https://github.com/hancyran/LiDAR-Diffusion """
class LidarDiffusionDecoder2(nn.Module):
    """ This one has the curve-wise blocks. It is the one they talk about in the LD paper. """
    def __init__(self, *, ch, out_ch, ch_mult, strides, num_res_blocks, attn_levels,
                 dropout=0.0, resamp_with_conv=True, in_channels, z_channels, give_pre_end=False,
                 tanh_out=False, use_linear_attn=False, attn_type="vanilla", use_mask=False,
                 **ignorekwargs):
        super().__init__()
        stride2kernel = {(2, 2): (3, 3), (1, 2): (1, 4)}
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = CircularConv2d(z_channels,
                                      block_in,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)

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

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            stride = tuple(strides[i_level - 1]) if i_level > 0 else None
            kernel = stride2kernel[stride] if stride is not None else (1, 4)
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock2(in_channels=block_in,
                                         out_channels=block_out,
                                         kernel_size=kernel,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if i_level in attn_levels:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if stride is not None:
                up.upsample = Upsample2(block_in, resamp_with_conv, stride)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = CircularConv2d(block_in,
                                       out_ch,
                                       kernel_size=(1, 4),
                                       stride=1,
                                       padding=(1, 2, 0, 0))

    def forward(self, z):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h