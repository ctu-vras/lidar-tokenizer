import numpy as np
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


import torch.nn.functional as F


from scipy.cluster.vq import kmeans2
from torch import nn
import torch.distributed.nn.functional
import torch.distributed

from typing import List, Optional

from torch.nn import Module
from torch import Tensor, int32

from einops import rearrange#, pack, unpack

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

# main class
""" 
For this thesis
Adapted from https://github.com/duchenzhuang/FSQ-pytorch/blob/main/quantizers/fsq.py
"""
class FSQ(Module):
    def __init__(
        self,
        levels: List[int] = [8,5,5,5],
        embed_dim: Optional[int] = None,    # keep this here
        rearrange: bool = False,
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        dead_limit: int = 256,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:len(levels)-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent = False)

        self.scale = scale
        self.rearrange = rearrange
        self.dead_limit = dead_limit

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out = False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width
    
    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width
    
    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)
    
    def indices_to_codes(
        self,
        indices: Tensor,
        project_out = True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes
    

    def get_codebook_entry(self, indices):
        original_shape = indices.shape
        indices = indices.reshape(original_shape[0],-1,1)
        codes = self.indices_to_codes(indices, project_out = True)
        codes = codes.reshape(original_shape[0], -1, original_shape[1], original_shape[2])
        return codes  
    

    def forward(self, z: Tensor, code_age: Tensor, code_usage: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """
        if self.rearrange:
            original_shape = z.shape
            z = z.reshape(z.shape[0], z.shape[1], -1)
            z = torch.swapaxes(z, 1, 2)        

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        # codebook metrics
        flat_indices = indices.reshape(-1)
        code_usage *= 0
        code_usage.index_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=code_usage.dtype))
        code_age += 1
        code_age[flat_indices.long()] = 0

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        if self.rearrange:
            out = torch.swapaxes(out, 1, 2)
            out = out.reshape(original_shape)

        return out, torch.Tensor([0]).mean(), indices


""" Adapted from https://github.com/hancyran/LiDAR-Diffusion """
class LidarDiffusionQuantizer(nn.Module):
    """ original name VectorQuantizer2 """
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_embed, embed_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_embed
        self.e_dim = embed_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.last_used = torch.zeros(self.n_e)
        self.dead_limit = 256

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = self.n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, code_age, code_usage, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        #bincount = min_encoding_indices.bincount()
        #code_uniformity = bincount.topk(10)[0].sum()/d.shape[0]

        code_usage *= 0
        code_usage.index_add_(0, min_encoding_indices, torch.ones_like(min_encoding_indices, dtype=code_usage.dtype))
        code_age += 1
        code_age[min_encoding_indices] = 0
        
        #used_code = torch.where(bincount > 0)

        #self.last_used += 1
        #self.last_used[used_code] = 0

        #print("\n")
        #print(used_code[0])
        #print("\n")
        #num_dead_code = (self.last_used > self.dead_limit).sum()
        #code_utilization = 1-num_dead_code/self.n_e

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

""" Adapted from https://github.com/myc634/UltraLiDAR_nusc_waymo """
class VectorQuantizer(nn.Module):
    def __init__(self, n_embed, embed_dim, beta, cosine_similarity=False, dead_limit=256, rearrange=False):
        super().__init__()
        self.n_e = n_embed
        self.e_dim = embed_dim
        self.beta = beta
        self.cosine_similarity = cosine_similarity
        self.dead_limit = dead_limit
        self.rearrange = rearrange

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.register_buffer("global_iter", torch.zeros(1))
        self.register_buffer("num_iter", torch.zeros(1))
        self.register_buffer("data_initialized", torch.zeros(1))
        self.register_buffer("reservoir", torch.zeros(self.n_e * 10, self.e_dim))

    def train_codebook(self, z, code_age, code_usage):
        assert z.shape[-1] == self.e_dim
        z_flattened = z.reshape(-1, self.e_dim)

        if self.cosine_similarity:
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)

        self.update_reservoir(z_flattened, code_age, code_usage)

        if self.cosine_similarity:
            min_encoding_indices = torch.matmul(z_flattened, F.normalize(self.embedding.weight, p=2, dim=-1).T).max(
                dim=-1
            )[1]
        else:
            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

            z_dist = torch.cdist(z_flattened, self.embedding.weight)
            min_encoding_indices = torch.argmin(z_dist, dim=1)
        
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        if self.cosine_similarity:
            z_q = F.normalize(z_q, p=2, dim=-1)
            z_norm = F.normalize(z, p=2, dim=-1)
            loss = self.beta * torch.mean(1 - (z_q.detach() * z_norm).sum(dim=-1)) + torch.mean(1 - (z_q * z_norm.detach()).sum(dim=-1)),
            
        else:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        if code_age is not None and code_usage is not None:
            code_idx = min_encoding_indices
            if torch.distributed.is_initialized():
                code_idx = torch.cat(torch.distributed.nn.functional.all_gather(code_idx))
            code_age += 1
            code_age[code_idx] = 0
            code_usage.index_add_(0, code_idx, torch.ones_like(code_idx, dtype=code_usage.dtype))


        return z_q, loss, min_encoding_indices



    def forward(self, z, code_age=None, code_usage=None):
        if self.rearrange:
            z = rearrange(z, 'b c h w -> b h w c').contiguous()
            z_rearranged = z.view(z.shape[0], z.shape[1] * z.shape[2], z.shape[3])
        else:
            z_rearranged = z
 
        z_q, loss, min_encoding_indices = self.train_codebook(z_rearranged, code_age, code_usage)
    
        if self.rearrange:
            z_q = z_q.view(z.shape)
            z_q = rearrange(z_q, 'b h w c -> b c h w ').contiguous()

        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, shape=None):
        # shape specifying (batch, height, width, channel)

        # get quantized latent vectors
        z_q = self.embedding(indices)

        #if shape is not None:
            #z_q = z_q.view(shape)
            # reshape back to match original input shape
            #z_q = z_q.permute(0, 3, 1, 2).contiguous()
        if self.cosine_similarity:
            z_q = F.normalize(z_q, p=2, dim=-1)
        z_q = z_q.reshape(indices.shape[0], indices.shape[1], -1)
        print(f"z_q {z_q.shape}")
        return z_q

    def update_reservoir(self, z, code_age, code_usage):
        if not (self.embedding.weight.requires_grad and self.training):
            return

        assert z.shape[-1] == self.e_dim
        z_flattened = z.reshape(-1, self.e_dim)

        rp = torch.randperm(z_flattened.size(0))
        num_sample: int = self.reservoir.shape[0] // 100  # pylint: disable=access-member-before-definition
        self.reservoir: torch.Tensor = torch.cat([self.reservoir[num_sample:], z_flattened[rp[:num_sample]].data])

        self.num_iter += 1
        self.global_iter += 1

        if ((code_age >= self.dead_limit).sum() / self.n_e) > 0.03 and (
            self.data_initialized.item() == 0 or self.num_iter.item() > 1000
        ):
            self.update_codebook(code_age, code_usage)
            if self.data_initialized.item() == 0:
                self.data_initialized.fill_(1)

            self.num_iter.fill_(0)

    def update_codebook(self, code_age, code_usage):
        if not (self.embedding.weight.requires_grad and self.training):
            return

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            live_code = self.embedding.weight[code_age < self.dead_limit].data
            live_code_num = live_code.shape[0]
            if self.cosine_similarity:
                live_code = F.normalize(live_code, p=2, dim=-1)

            all_z = torch.cat([self.reservoir, live_code])
            rp = torch.randperm(all_z.shape[0])
            all_z = all_z[rp]

            init = torch.cat(
                [live_code, self.reservoir[torch.randperm(self.reservoir.shape[0])[: (self.n_e - live_code_num)]]]
            )
            init = init.data.cpu().numpy()
            print(
                "running kmeans!!", self.n_e, live_code_num, self.data_initialized.item()
            )  # data driven initialization for the embeddings
            centroid, assignment = kmeans2(
                all_z.cpu().numpy(),
                init,
                minit="matrix",
                iter=50,
            )
            z_dist = (all_z - torch.from_numpy(centroid[assignment]).to(all_z.device)).norm(dim=1).sum().item()
            self.embedding.weight.data = torch.from_numpy(centroid).to(self.embedding.weight.device)

            print("finish kmeans", z_dist)

        if torch.distributed.is_initialized():
            torch.distributed.nn.functional.broadcast(self.embedding.weight, src=0)

        code_age.fill_(0)
        code_usage.fill_(0)
        # self.data_initialized.fill_(1)