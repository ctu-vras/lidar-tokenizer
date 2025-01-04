import numpy as np
import torch
import pytorch_lightning as pl
from copy import deepcopy
import torch.nn.functional as F
from contextlib import contextmanager
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange
from functools import reduce
from operator import mul
from time import time
from torch import nn

try:
    from data_managment.projections import pcd2range, process_scan, range2pcd
    from data_managment.voxelize import Voxelizer, LogVoxelizer
    from model_managment.helpers import instantiate_from_config, pad_tensor
    from model_managment.encoders import LidarDiffusionEncoder, LidarDiffusionEncoder2, VQEncoder
    from model_managment.decoders import LidarDiffusionDecoder, LidarDiffusionDecoder2, VQDecoder
    from model_managment.codebooks import LidarDiffusionQuantizer, FSQ
    #from model_managment.ultralidar import *
except Exception as e:
    print(e)
    print("\nTRYING WITH RELATIVE IMPORTS, BUT IF THE ISSUE WAS SOMETHING ELSE THEN YOU ARE PROBABLY MISSING A PACKAGE\n")
    from ..data_managment.projections import pcd2range, process_scan, range2pcd
    from ..data_managment.voxelize import Voxelizer, LogVoxelizer
    from ..model_managment.helpers import instantiate_from_config, pad_tensor
    from ..model_managment.encoders import LidarDiffusionEncoder, LidarDiffusionEncoder2, VQEncoder
    from ..model_managment.decoders import LidarDiffusionDecoder, LidarDiffusionDecoder2, VQDecoder
    from ..model_managment.codebooks import LidarDiffusionQuantizer, FSQ
    #from ..model_managment.ultralidar import *

from functools import partial
import omegaconf
from torchmetrics.regression import MeanSquaredError,MeanAbsoluteError
from collections import defaultdict
from torch.optim.lr_scheduler import CosineAnnealingLR

""" ------------------------------------------------------- """
""" ---------------------UltraLidar MODEL------------------ """
""" ------------------------------------------------------- """

class UltraLidarModel(pl.LightningModule):
    # original name VQModel
    def __init__(self,
                voxelizer,
                codebook,
                patch_size = 8,
                img_size = None,
                pos_weight = 1.,
                bias_init = -3.,
                sigma = 0.5,
            ):
        super().__init__()

        if 'n_embed' in codebook['params']:
            self.n_embed = codebook['params']['n_embed']
        else:
            self.n_embed = reduce(mul, codebook['params']['levels'])

        if 'embed_dim' in codebook['params']:
            self.embed_dim = codebook['params']['embed_dim']
        else:
            self.embed_dim = len(codebook['params']['levels'])
        
        self.bias_init = bias_init
        self.sigma = sigma

        if isinstance(img_size, tuple):
            pass
        elif img_size is None:
            img_size = (round((voxelizer['params']['y_max'] - voxelizer['params']['y_min'])/voxelizer['params']['step']),
                        round((voxelizer['params']['x_max'] - voxelizer['params']['x_min'])/voxelizer['params']['step']))
        elif isinstance(img_size, int):
            img_size = (img_size,img_size)
        elif isinstance(img_size, list) or isinstance(img_size, omegaconf.listconfig.ListConfig):
            img_size = tuple(img_size)
        else:
            raise TypeError(f"img_size needs to be None,tuple,int or list or omegaconf.listconfig.ListConfig not {type(img_size)}")
        
        """ self.voxelizer = Voxelizer(xmin,
                                   xmax,
                                   ymin,
                                   ymax,
                                   pixel_size,
                                   zmin,
                                   zmax,
                                   zstep)
        
        self.voxelizer = LogVoxelizer(x_min = xmin,
                                   x_max = xmax,
                                   step = pixel_size,
                                   z_min = zmin,
                                   z_max = zmax,
                                   z_step = zstep) """
        self.voxelizer = instantiate_from_config(voxelizer)

        self.in_chans = self.voxelizer.z_depth

        self.lidar_encoder = VQEncoder(
            patch_size=patch_size,
            in_chans=self.in_chans,
            img_size=img_size,
            codebook_dim=self.embed_dim)

        self.pre_quant = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.LayerNorm(self.embed_dim))
        
        self.vector_quantizer = instantiate_from_config(codebook)
        
        self.register_buffer("code_age", torch.zeros(self.n_embed) * 10000)
        self.register_buffer("code_usage", torch.zeros(self.n_embed))

        if isinstance(img_size, tuple):
            img_size_dec = img_size
            num_patches_dec = int(img_size[0]*img_size[1]/patch_size**2)
        else:
            img_size_dec = (img_size, img_size) 
            num_patches_dec = int(img_size*img_size/patch_size**2)

        self.lidar_decoder = VQDecoder(
            img_size=img_size_dec,
            patch_size=patch_size,
            in_chans=self.in_chans,
            num_patches=num_patches_dec,
            codebook_dim=self.embed_dim,
            bias_init = self.bias_init)
        
        self.pos_weight = pos_weight

    def process_output(self, model_out):
        pred_points = torch.where(model_out.sigmoid() >= self.sigma)
        model_out *= 0
        model_out[pred_points[0],pred_points[1],pred_points[2],pred_points[3]] = 1

        all_pcd_out = []

        for i in range(model_out.shape[0]):
            all_pcd_out.append(self.voxelizer.voxels2points(model_out[i]))

        return all_pcd_out
    
    def decode_code(self, code):
        quant = self.vector_quantizer.get_codebook_entry(code)
        rec = self.lidar_decoder(quant)
        return rec

    def encode(self, voxels):
        lidar_feats = self.lidar_encoder(voxels)
        feats = self.pre_quant(lidar_feats)
        lidar_quant, emb_loss, ind = self.vector_quantizer(feats, self.code_age, self.code_usage)
        return lidar_quant, emb_loss, ind

    def forward(self, voxels, split="train"):
        lidar_feats = self.lidar_encoder(voxels)
        feats = self.pre_quant(lidar_feats)
        lidar_quant, emb_loss, ind = self.vector_quantizer(feats, self.code_age, self.code_usage)
        #emb_loss = emb_loss.to(self.device)
        lidar_rec = self.lidar_decoder(lidar_quant)

        #pos_weight = torch.ones((self.in_chans,1,1), device=self.device) * self.pos_weight
        lidar_rec_loss = (F.binary_cross_entropy_with_logits(lidar_rec, voxels, reduction="none") * 100).mean()
        lidar_rec_prob = lidar_rec.sigmoid().detach()
        lidar_rec_diff = (lidar_rec_prob - voxels).abs().sum() / voxels.shape[0]
        lidar_rec_iou = ((lidar_rec_prob >= self.sigma) & (voxels >= self.sigma)).sum() / (
            (lidar_rec_prob >= self.sigma) | (voxels >= self.sigma)
            ).sum()
        code_util = (self.code_age < self.vector_quantizer.dead_limit).sum() / self.code_age.numel()
        code_uniformity = self.code_usage.topk(10)[0].sum() / self.code_usage.sum()

        loss = emb_loss * 10 + lidar_rec_loss

        out = dict()
        out.update(
            {
                f"{split}/loss": loss,
                f"{split}/loss_lidar_rec": lidar_rec_loss,
                f"{split}/loss_emb": emb_loss * 10,
                f"{split}/lidar_rec_diff": lidar_rec_diff,
                f"{split}/lidar_rec_iou": lidar_rec_iou,
                f"{split}/code_util": code_util,
                f"{split}/code_uniformity": code_uniformity,
            }
        )
        if split == "test":
            return out, lidar_rec, ind
        else:
            return out, lidar_rec
        
    def training_step(self, batch, batch_idx):
        out, _ = self(batch,"train")

        self.log_dict(out, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return out["train/loss"]

    def validation_step(self, batch, batch_idx):
        out, _  = self(batch,"val")

        self.log_dict(out, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return out["val/loss"]
    
    """ def voxels2points(self, voxels):

        non_zero_indices = torch.nonzero(voxels)
        xy = (non_zero_indices[:, [3,2]] * self.voxelizer.step) + torch.tensor([self.voxelizer.x_min, self.voxelizer.y_min], device=self.device)
        z = (non_zero_indices[:, 1] * self.voxelizer.z_step) + self.voxelizer.z_min
        xyz = torch.cat([xy, z.unsqueeze(1)], dim=1)

        return xyz """
    
    def predict_step(self, batch, batch_idx):
        voxels = batch['voxels']
        pcd_in = batch['pointcloud']
        
        out, lidar_rec, codes = self(voxels, split='test')
        pred_points = torch.where(lidar_rec.sigmoid() >= self.sigma)
        lidar_rec *= 0
        lidar_rec[pred_points[0],pred_points[1],pred_points[2],pred_points[3]] = 1

        all_pcd_out = []
        all_pcd_voxelized_in = []

        for i in range(lidar_rec.shape[0]):
            all_pcd_out.append(self.voxelizer.voxels2points(lidar_rec[i]))
            all_pcd_voxelized_in.append(self.voxelizer.voxels2points(voxels[i]))

        return out, pcd_in, all_pcd_out, all_pcd_voxelized_in, codes

    def configure_optimizers(self):
        """ def get_param_groups():
            param_groups = []
            for name, param in self.named_parameters():
                if "absolute_pos_embed" in name or "relative_position_bias_table" in name or "norm" in name or "embedding" in name:
                    param_groups.append({"params": param, "weight_decay": 0.0})
                elif "img_backbone" in name:
                    param_groups.append({"params": param, "lr": self.lr * 0.1, "weight_decay": 0.0001 * 0.001})
                else:
                    param_groups.append({"params": param})
            return param_groups

        opt = torch.optim.AdamW(
            get_param_groups(),
            betas=(0.9, 0.95),
            weight_decay=0.0001,
            lr=self.lr
        )

        return {
            'optimizer': opt
        }  """
    
        """
        opt = torch.optim.AdamW(self.parameters(), weight_decay=0.005, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95) 
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',   
                'frequency': 300      
            }
        } 
        """

        def get_param_groups():
            param_groups = []
            for name, param in self.named_parameters():
                if "absolute_pos_embed" in name or "relative_position_bias_table" in name or "norm" in name or "embedding" in name:
                    param_groups.append({"params": param, "weight_decay": 0.0})
                elif "img_backbone" in name:
                    param_groups.append({"params": param, "lr": self.lr * 0.1, "weight_decay": 0.0001 * 0.001})
                else:
                    param_groups.append({"params": param})
            return param_groups

        opt = torch.optim.AdamW(
            get_param_groups(),
            betas=(0.9, 0.95),
            weight_decay=0.0001,
            lr=self.lr
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        opt,
                        max_lr = self.lr,
                        total_steps = self.trainer.max_steps,
                        pct_start=0.3,
                        anneal_strategy="cos",
                        div_factor=10.0,
                        final_div_factor=100.0,)

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',   # or 'step' for per-step updates
                'frequency': 1         # how often to apply the scheduler
            }
        }

        """ #lr_lambda_fn = LRLambda(self.trainer.max_epochs)
        #scheduler = LambdaLR(opt, lr_lambda_fn)

        CosineAnnealingLR(opt,)

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
            }
        }  """


""" ------------------------------------------------------- """
""" ----------------FULL RANGE IMAGE MODEL------------------ """
""" ------------------------------------------------------- """
""" Not used in the thesis. This one is the one from the LD paper and it is of
the GAN architecture. We omit the discriminator in our experiments and for that
we define another class lower.

This model is NOT up to date, but it is left here if anyone is interested in using
the discriminator. But it would require adjustments to match the rest of the pipeline... """

class LidarDiffusionModel(pl.LightningModule):
    # original name VQModel
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 lossconfig=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="input2d",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 use_ema=False,
                 lib_name='ldm',
                 use_mask=False,
                 **kwargs
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.use_mask = use_mask
        self.fov = lossconfig['params']['dataset_config']['fov']                    # needed for inference...
        self.depth_range = lossconfig['params']['dataset_config']['depth_range']    # needed for inference...        
        self.log_scale = lossconfig['params']['dataset_config']['log_scale']        # needed for inference...

        if self.log_scale:
            self.depth_scale = lossconfig['params']['dataset_config']['depth_scale']
        else:
            self.depth_scale = 2**lossconfig['params']['dataset_config']['depth_scale']

        self.encoder = LidarDiffusionEncoder(**ddconfig)
        self.decoder = LidarDiffusionDecoder(**ddconfig)
        if lossconfig is not None:
            self.loss = instantiate_from_config(lossconfig)
        self.quantize = LidarDiffusionQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        """ self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.") """

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

        self.test_img_mse = MeanSquaredError()
        self.test_img_mae = MeanAbsoluteError()
        self.test_points_mse = MeanSquaredError()
        self.test_points_mae = MeanAbsoluteError()
        self.test_points_clamped_mse = MeanSquaredError()
        self.test_points_clamped_mae = MeanAbsoluteError()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        # if len(x.shape) == 3:
        #     x = x[..., None]

        if self.batch_resize_range is not None:
            print("EHhhhh")
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size + 16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def get_mask(self, batch):
        mask = batch['mask']
        # if len(mask.shape) == 3:
        #     mask = mask[..., None]
        return mask
    
    def predict_step(self, batch, batch_idx):
        if not self.use_mask:
            raise NotImplementedError("Now is time to check this is correctly implemented for NO MASK!! :) Remember?")
        
        x = self.get_input(batch, self.image_key)
        m = self.get_mask(batch) if self.use_mask else None

        #x_original = torch.clone(x)
        
        model_out, _, _ = self(x, return_pred_indices=True)
        
        x_rec = model_out[:,0]
        m_rec = model_out[:,1] if self.use_mask else None

        out = defaultdict(list)

        x = x.cpu()
        m = m.cpu()
        x_rec = x_rec.cpu()
        m_rec = m_rec.cpu()

        for i in range(x_rec.shape[0]): # for item in batch
            x_img = x[i].numpy()
            #x_img = x_img[0:1,10:11,300:700]
            pcd,_,_ = range2pcd(x_img, self.fov, self.depth_range, self.depth_scale, self.log_scale, label=None, color=None)
            pcd_metrics,_,_ = range2pcd(x_img, self.fov, self.depth_range, self.depth_scale, self.log_scale, label=None, color=None, clamp=True)
            x_img = x_img[0]

            x_rec_img = x_rec[i:i+1].numpy()
            #x_rec_img = x_rec_img[0:1,10:11,300:700]
            pcd_rec,_,_ = range2pcd(x_rec_img, self.fov, self.depth_range, self.depth_scale, self.log_scale, label=None, color=None)
            pcd_rec_metrics,_,_ = range2pcd(x_rec_img, self.fov, self.depth_range, self.depth_scale, self.log_scale, label=None, color=None, clamp=True)
            #pcd_rec_clamped_metrics,_,_ = range2pcd(x_rec_img, self.fov, self.depth_range, self.depth_scale, self.log_scale, label=None, color=None, mask_invalid=False)

            x_rec_masked = x_rec_img.copy()
            #m_rec = m_rec[0:1,10:11,300:700]
            x_rec_masked[0:1, np.argwhere(m_rec[i]<0)[0], np.argwhere(m_rec[i]<0)[1]] = -1
            pcd_rec_masked,_,_ = range2pcd(x_rec_masked, self.fov, self.depth_range, self.depth_scale, self.log_scale, label=None, color=None)

            test_img_mse = self.test_img_mse(torch.from_numpy(x_rec_img[0]), torch.from_numpy(x_img)).cpu()
            test_img_mae = self.test_img_mae(torch.from_numpy(x_rec_img[0]), torch.from_numpy(x_img)).cpu()
            test_points_mse = self.test_points_mse(torch.from_numpy(pcd_rec_metrics), torch.from_numpy(pcd_metrics)).cpu()
            test_points_mae = self.test_points_mae(torch.from_numpy(pcd_rec_metrics), torch.from_numpy(pcd_metrics)).cpu()
            #test_points_clamped_mse = self.test_points_mse(torch.from_numpy(np.clip(pcd_rec_metrics, -70., 70.)), torch.from_numpy(pcd_metrics))
            #test_points_clamped_mae = self.test_points_mae(torch.from_numpy(np.clip(pcd_rec_metrics, -70., 70.)), torch.from_numpy(pcd_metrics))


            if not self.metrics_only:
                out['x'].append(x_img)
                out['pcd'].append(pcd)
                #out['m'] = m[i,0,10:11,300:700]
                out['m'].append(m[i,0])
                out['x_rec'].append(x_rec_img[0])
                out['pcd_rec'].append(pcd_rec)
                out['pcd_rec_masked'].append(pcd_rec_masked)
                out['m_rec'].append(m_rec[i])
            out['test_img_mse'].append(test_img_mse)
            out['test_img_mae'].append(test_img_mae)
            out['test_points_mse'].append(test_points_mse)
            out['test_points_mae'].append(test_points_mae)
            #out['test_points_clamped_mse'] = test_points_clamped_mse
            #out['test_points_clamped_mae'] = test_points_clamped_mae

        return out

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        m = self.get_mask(batch) if self.use_mask else None
        x_rec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencoder
            aeloss, log_dict_ae = self.loss(qloss, x, x_rec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=None, masks=m)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

            #self.train_mse(x_rec, x)
            #self.train_mae(x_rec, x)

            #pcd_rec,_,_ = range2pcd(x_rec[i:i+1].numpy(), self.fov, self.depth_range, self.depth_scale, self.log_scale, label=None, color=None)
            #pcd,_,_ = range2pcd(x[i].numpy(), self.fov, self.depth_range, self.depth_scale, self.log_scale, label=None, color=None)

            #self.log('train/mse', self.train_mse, on_step=True, on_epoch=False)
            #self.log('train/mae', self.train_mae, on_step=True, on_epoch=False)

            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, x_rec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train",
                                                masks=m)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        m = self.get_mask(batch) if self.use_mask else None
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val" + suffix,
                                        predicted_indices=None,
                                        masks=m
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val" + suffix,
                                            predicted_indices=None,
                                            masks=m
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.lr
        lr_g = self.lr_g_factor * self.lr
        # print("lr_d", lr_d)
        # print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if self.use_mask:
            mask = xrec[:, 1:2] < 0.
            xrec = xrec[:, 0:1]
            xrec[mask] = -1.
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x
    

""" This is the model used in the experiments. """

class LidarDiffusionModelNoGan(pl.LightningModule):
    # original name VQModel
    def __init__(self,
                 encoder_decoder,
                 codebook,
                 lossconfig=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="input2d",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 use_ema=False,
                 lib_name='ldm',
                 use_mask=False,
                 **kwargs
                 ):
        super().__init__()

        if 'n_embed' in codebook['params']:
            self.n_embed = codebook['params']['n_embed']
        else:
            self.n_embed = reduce(mul, codebook['params']['levels'])
        self.embed_dim = codebook['params']['embed_dim']
        self.z_channels = encoder_decoder['params']['z_channels']
        self.image_key = image_key
        self.use_mask = use_mask
        self.fov = lossconfig['params']['dataset_config']['fov']                    # needed for inference...
        self.depth_range = lossconfig['params']['dataset_config']['depth_range']    # needed for inference...
        self.log_scale = lossconfig['params']['dataset_config']['log_scale']        # needed for inference...
        self.sensor_pos = lossconfig['params']['dataset_config']['sensor_pos']
        self.size = lossconfig['params']['dataset_config']['size']

        if 't_ratio' in lossconfig['params']['dataset_config']:
            self.t_ratio = lossconfig['params']['dataset_config']['t_ratio']
            self.b_ratio = lossconfig['params']['dataset_config']['b_ratio']
            self.l_ratio = lossconfig['params']['dataset_config']['l_ratio']
            self.r_ratio = lossconfig['params']['dataset_config']['r_ratio']

        if self.log_scale:
            self.depth_scale = lossconfig['params']['dataset_config']['depth_scale']
        else:
            self.depth_scale = 2**lossconfig['params']['dataset_config']['depth_scale']

        encoder_config = encoder_decoder
        encoder_config['target'] = encoder_decoder['encoder_target']
        self.encoder = instantiate_from_config(encoder_config) #LidarDiffusionEncoder(**ddconfig)

        decoder_config = encoder_decoder
        decoder_config['target'] = encoder_decoder['decoder_target']
        self.decoder = instantiate_from_config(decoder_config) #LidarDiffusionDecoder(**ddconfig)

        if lossconfig is not None:
            self.loss = instantiate_from_config(lossconfig)
        self.quantize = instantiate_from_config(codebook)#LidarDiffusionQuantizer(n_embed, embed_dim, beta=0.25,
                        #                remap=remap,
                        #                sane_index_shape=sane_index_shape)

        if self.n_embed is not None:
            self.register_buffer("code_age", torch.zeros(self.n_embed) * 10000)
            self.register_buffer("code_usage", torch.zeros(self.n_embed))
        else:
            self.register_buffer("code_age", None)
            self.register_buffer("code_usage", None)

        self.quant_conv = torch.nn.Conv2d(self.z_channels, self.embed_dim, 1) # just to get the correct number of channels
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, self.z_channels, 1)


        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")


        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

        self.test_img_mse = MeanSquaredError()
        self.test_img_mae = MeanAbsoluteError()
        self.test_points_mse = MeanSquaredError()
        self.test_points_mae = MeanAbsoluteError()
        self.test_points_clamped_mse = MeanSquaredError()
        self.test_points_clamped_mae = MeanAbsoluteError()
        self.metrics_only = False

    def encode(self, x):
        #t = time()
        h = self.encoder(x)
        #print(f"encoder {time()-t}")
        #t = time()
        h = self.quant_conv(h)
        #print(f"conv {time()-t}")
        #t = time()
        if self.code_age is None and self.code_usage is None:
            quant, emb_loss, ind = self.quantize(h)
        else:
            quant, emb_loss, ind = self.quantize(h, self.code_age, self.code_usage)
        #print(f"quant {time()-t}")

        #print(quant.shape)

        return quant, emb_loss, ind

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.get_codebook_entry(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, ind = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        # if len(x.shape) == 3:
        #     x = x[..., None]

        if self.batch_resize_range is not None:
            print("EHhhhh")
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size + 16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def get_mask(self, batch):
        mask = batch['mask']
        # if len(mask.shape) == 3:
        #     mask = mask[..., None]
        return mask

    def process_output(self, model_out):
        x_rec = model_out[:,0:1]
        m_rec = model_out[:,1:2]

        x_rec = x_rec.cpu()
        m_rec = m_rec.cpu()

        ratios = (int(round(self.t_ratio*self.size[0])),
                  int(round(self.b_ratio*self.size[0])),
                  int(round(self.l_ratio*self.size[1])),
                  int(round(self.r_ratio*self.size[1])))
        x_rec_padded = pad_tensor(x_rec, self.size[0], self.size[1], ratios)
        m_rec_padded = pad_tensor(m_rec, self.size[0], self.size[1], ratios)[:,0]
        pcd_recs = []
        for i in range(x_rec_padded.shape[0]):
            x_rec_masked = x_rec_padded[i].numpy()
            x_rec_masked[0:1, np.argwhere(m_rec_padded[i]<0)[0], np.argwhere(m_rec_padded[i]<0)[1]] = -1
            pcd_rec_masked,_,_ = range2pcd(x_rec_masked, self.fov, self.depth_range, self.depth_scale, self.log_scale, label=None, color=None)
            pcd_rec_masked += self.sensor_pos
            pcd_rec_masked = torch.from_numpy(pcd_rec_masked)
            pcd_recs.append(pcd_rec_masked)

        # return images without padding
        x_rec_masked_no_padding = x_rec
        x_rec_masked_no_padding[m_rec<0] = -1

        return x_rec_masked_no_padding, pcd_recs
       
    def predict_step(self, batch, batch_idx):       
        x = self.get_input(batch, self.image_key)
        m = self.get_mask(batch) if self.use_mask else None

        model_out, _, ind = self(x, return_pred_indices=True)
        ind = ind.reshape(-1)
        
        x_rec = model_out[:,0:1]
        m_rec = model_out[:,1:2] if self.use_mask else None

        out = defaultdict(list)

        x = x.cpu()
        m = m.cpu()[:,0]
        x_rec = x_rec.cpu()
        m_rec = m_rec.cpu()[:,0]

        if "ratios" in batch:
            h,w,ratios = batch['ratios']

            # Pad in case of FrontCam
            # Assume all h and w are the same in the batch
            if isinstance(h, torch.Tensor):
                h = h[0]
                w = w[0]

            if x.shape[-2] != h or x.shape[-1] != w:
                x = pad_tensor(x, h, w, ratios)
                m = pad_tensor(m, h, w, ratios)[:,0]
                x_rec = pad_tensor(x_rec, h, w, ratios)
                m_rec = pad_tensor(m_rec, h, w, ratios)[:,0]
            
            out['ratios'] = ratios

        for i in range(x_rec.shape[0]): # for item in batch
            x_img = x[i].numpy()
            pcd,_,_ = range2pcd(x_img, self.fov, self.depth_range, self.depth_scale, self.log_scale, label=None, color=None)
            pcd += self.sensor_pos

            x_rec_img = x_rec[i].numpy()
            pcd_rec,_,_ = range2pcd(x_rec_img, self.fov, self.depth_range, self.depth_scale, self.log_scale, label=None, color=None)
            pcd_rec += self.sensor_pos

            x_rec_masked = x_rec_img.copy()
            x_rec_masked[0:1, np.argwhere(m_rec[i]<0)[0], np.argwhere(m_rec[i]<0)[1]] = -1
            pcd_rec_masked,_,_ = range2pcd(x_rec_masked, self.fov, self.depth_range, self.depth_scale, self.log_scale, label=None, color=None)
            pcd_rec_masked += self.sensor_pos

            x_rec_ground_truth_masked = x_rec_img.copy()
            pcd_rec_ground_truth_masked,_,_ = range2pcd(x_rec_ground_truth_masked, self.fov, self.depth_range, self.depth_scale,
                                                        self.log_scale, label=None, color=None, clamp=True, visualization_mask=m[i])
            pcd_rec_ground_truth_masked += self.sensor_pos

            x_rec_ground_truth_masked[0:1, np.argwhere(m[i]<0)[0], np.argwhere(m[i]<0)[1]] = -1
            pcd_ground_truth_masked,_,_ = range2pcd(x_img, self.fov, self.depth_range, self.depth_scale,
                                                    self.log_scale, label=None, color=None, clamp=True, visualization_mask=m[i])
            pcd_ground_truth_masked += self.sensor_pos

            test_img_mse = self.test_img_mse(torch.from_numpy(x_rec_ground_truth_masked[0]), torch.from_numpy(x_img[0])).cpu()
            test_img_mae = self.test_img_mae(torch.from_numpy(x_rec_ground_truth_masked[0]), torch.from_numpy(x_img[0])).cpu()
            test_points_mse = self.test_points_mse(torch.from_numpy(pcd_rec_ground_truth_masked), torch.from_numpy(pcd_ground_truth_masked)).cpu()
            test_points_mae = self.test_points_mae(torch.from_numpy(pcd_rec_ground_truth_masked), torch.from_numpy(pcd_ground_truth_masked)).cpu()

            if not self.metrics_only:
                out['pcd'].append(pcd)
                out['pcd_rec'].append(pcd_rec)
                out['pcd_rec_masked'].append(pcd_rec_masked)
                out['pcd_rec_ground_truth_masked'].append(pcd_rec_ground_truth_masked)
            out['x'].append(x_img[0])
            out['m'].append(m[i])
            out['x_rec'].append(x_rec_img[0])
            out['m_rec'].append(m_rec[i])
            out['ind'].append(ind)
            out['test_img_mse'].append(test_img_mse)
            out['test_img_mae'].append(test_img_mae)
            out['test_points_mse'].append(test_points_mse)
            out['test_points_mae'].append(test_points_mae)

        return out

    def training_step(self, batch, batch_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        m = self.get_mask(batch) if self.use_mask else None
        if 'ratios' in batch:
            ratios = batch['ratios'] 
        else:
            ratios = None
            
        x_rec, qloss, ind = self(x, return_pred_indices=True)

        aeloss, log_dict_ae = self.loss(qloss, x, x_rec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train",
                                        predicted_indices=None, masks=m, mask_invalid_in_loss=True,
                                        ratios=ratios)

        if self.code_age is not None and self.code_usage is not None:
            code_util = (self.code_age < self.quantize.dead_limit).sum() / self.code_age.numel()
            code_uniformity = self.code_usage.topk(10)[0].sum() / self.code_usage.sum()
        
            self.log(f"train/code_uniformity", code_uniformity,
                    prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"train/code_util", code_util,
                    prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        points_mse = log_dict_ae[f"train/points_mse"]
        self.log(f"train/points_mse", points_mse,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae[f"train/points_mse"]
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return aeloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        m = self.get_mask(batch) if self.use_mask else None
        if 'ratios' in batch:
            ratios = batch['ratios'] 
        else:
            ratios = None

        xrec, qloss, ind = self(x, return_pred_indices=True)
        
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val" + suffix,
                                        predicted_indices=None,
                                        masks=m, mask_invalid_in_loss=True,
                                        ratios=ratios)

        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        points_mse = log_dict_ae[f"val{suffix}/points_mse"]

        self.log(f"val{suffix}/points_mse", points_mse,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        self.log(f"val{suffix}/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        del log_dict_ae[f"val{suffix}/rec_loss"], log_dict_ae[f"val{suffix}/points_mse"]
        self.log_dict(log_dict_ae)

        return self.log_dict

    def configure_optimizers(self):

        opt = torch.optim.AdamW(list(self.parameters()),
                                lr=self.lr,
                                weight_decay=0.001,
                                betas=(0.5, 0.9))
        
        """ scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',   
                'frequency': 700      
            }
        } """
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        opt,
                        max_lr = self.lr,
                        total_steps = self.trainer.max_steps,
                        pct_start=0.3,
                        anneal_strategy="cos",
                        div_factor=10.0,
                        final_div_factor=100.0,)

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',   # or 'step' for per-step updates
                'frequency': 1         # how often to apply the scheduler
            }
        }

    def get_last_layer(self):
        return self.decoder.conv_out.weight
