# Trainer for MaskGIT
import os
import random
import time
import math

import numpy as np
from tqdm import tqdm
from collections import deque
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP

from Trainer.trainer import Trainer
from Network.transformer import MaskTransformer, MaskTransformerImg

from Network.Taming.models.vqgan import VQModel
from LiDAR_Tokenizer.data_managment.preprocess_frontcam import preprocess_range, preprocess_voxel
from LiDAR_Tokenizer.model_managment.helpers import load_model, pcd_to_plot_image

import yaml
import matplotlib.pyplot as plt

from PIL import Image
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

class MaskGIT(Trainer):

    def __init__(self, args):
        """ Initialization of the model (VQGAN and Masked Transformer), optimizer, criterion, etc."""
        super().__init__(args)
        self.args = args                                                        # Main argument see main.py
        self.scaler = torch.cuda.amp.GradScaler()                               # Init Scaler for multi GPUs
        self.ae = self.get_network("autoencoder")

        self.img_only = self.args.img_only

        if self.img_only:
            self.lidar_tokenizer = None
            self.lidar_config = None
        else:
            self.lidar_tokenizer = self.get_network("lidar_tokenizer")
            with open(self.args.lidar_config, 'r') as file:
                lidar_config = yaml.safe_load(file)
            self.lidar_config = lidar_config

        self.codebook_size_img = self.args.codebook_size_img #self.ae.n_embed  
        self.codebook_size_lidar = self.args.codebook_size_lidar 
        
        self.mode = args.mode

        if self.img_only:
            self.codebook_size = self.codebook_size_img  
        else:
            self.codebook_size = self.codebook_size_img + self.codebook_size_lidar   
        
        self.patch_size = self.args.img_size // 2**(self.ae.encoder.num_resolutions-1)     # Load VQGAN

        self.num_tokens_img = self.patch_size * self.patch_size
        self.num_tokens_lidar = self.args.num_tokens_lidar

        if self.img_only:
            self.num_tokens = self.num_tokens_img
        else:
            self.num_tokens = self.num_tokens_img + self.num_tokens_lidar
            
        print("Acquired codebook size:", self.codebook_size)   
        self.vit = self.get_network("vit")                                      # Load Masked Bidirectional Transformer  

        self.criterion = self.get_loss("cross_entropy", label_smoothing=0.1)    # Get cross entropy loss

        if self.args.onecycle:
            self.optim, self.lr_scheduler = self.get_optim(self.vit, self.args.lr, num_train_samples=60960, onecycle=self.args.onecycle, epochs=self.args.epoch, bs=self.args.bsize*self.args.num_gpus*self.args.grad_cum, betas=(0.9, 0.96))  # Get Adam Optimizer with weight decay
        else:    
            self.optim = self.get_optim(self.vit, self.args.lr, onecycle=self.args.onecycle, epochs=self.args.epoch, bs=self.args.bsize*self.args.num_gpus, betas=(0.9, 0.96))  # Get Adam Optimizer with weight decay

        # Load data if aim to train or test the model
        if not self.args.debug:
            self.train_data, self.test_data = self.get_data()

        # Initialize evaluation object if testing
        if self.args.test_only:
            from Metrics.sample_and_eval import SampleAndEval
            self.sae = SampleAndEval(device=self.args.device, num_images=19400, num_classes=0)

    def get_network(self, archi):
        """ return the network, load checkpoint if self.args.resume == True
            :param
                archi -> str: vit|autoencoder, the architecture to load
            :return
                model -> nn.Module: the network
        """
        if archi == "vit":
            if self.img_only:
                    model = MaskTransformerImg(
                    img_size=self.args.img_size, hidden_dim=768, codebook_size_img=self.codebook_size_img,
                    num_tokens_img = self.num_tokens_img, depth=24, heads=16, mlp_dim=3072, dropout=0.1     # Small
                    # img_size=self.args.img_size, hidden_dim=1024, codebook_size=1024, depth=32, heads=16, mlp_dim=3072, dropout=0.1  # Big
                    # img_size=self.args.img_size, hidden_dim=1024, codebook_size=1024, depth=48, heads=16, mlp_dim=3072, dropout=0.1  # Huge
                )
            else:
                model = MaskTransformer(
                    img_size=self.args.img_size, hidden_dim=768, codebook_size_img=self.codebook_size_img, codebook_size_lidar=self.codebook_size_lidar,
                    num_tokens_img = self.num_tokens_img, num_tokens_lidar = self.num_tokens_lidar, depth=24, heads=16, mlp_dim=3072, dropout=0.1     # Small
                    # img_size=self.args.img_size, hidden_dim=1024, codebook_size=1024, depth=32, heads=16, mlp_dim=3072, dropout=0.1  # Big
                    # img_size=self.args.img_size, hidden_dim=1024, codebook_size=1024, depth=48, heads=16, mlp_dim=3072, dropout=0.1  # Huge
                )

            if self.args.resume:
                ckpt = self.args.vit_folder
                ckpt += "current.pth" if os.path.isdir(self.args.vit_folder) else ""
                if self.args.is_master:
                    print("load ckpt from:", ckpt)
                # Read checkpoint file
                checkpoint = torch.load(ckpt, map_location='cpu')
                # Update the current epoch and iteration
                self.args.iter += checkpoint['iter']
                self.args.global_epoch += checkpoint['global_epoch']
                # Load network
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            model = model.to(self.args.device)
            if self.args.is_multi_gpus:  # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])

        elif archi == "autoencoder":
            # Load config
            config = OmegaConf.load(self.args.vqgan_folder + "model.yaml")
            model = VQModel(**config.model.params)
            checkpoint = torch.load(self.args.vqgan_folder + "last.ckpt", map_location="cpu")["state_dict"]
            # Load network
            model.load_state_dict(checkpoint, strict=False)
            model = model.eval()
            model = model.to(self.args.device)
            

            if self.args.is_multi_gpus: # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])
                model = model.module
        elif archi == "lidar_tokenizer":
            model = load_model(self.args.lidar_config)
            checkpoint = torch.load(self.args.lidar_ckpt, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            model = model.to(self.args.device)
        else:
            model = None

        if self.args.is_master:
            print(f"Size of model {archi}: "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

        return model

    @staticmethod
    def get_mask_code(code, mode="arccos", value=None, codebook_size=256):
        """ Replace the code token by *value* according the the *mode* scheduler
           :param
            code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
            mode  -> str:                the rate of value to mask
            value -> int:                mask the code by the value
           :return
            masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
            mask        -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask
        """
        r = torch.rand(code.size(0))
        if mode == "linear":                # linear scheduler
            val_to_mask = r
        elif mode == "square":              # square scheduler
            val_to_mask = (r ** 2)
        elif mode == "cosine":              # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":              # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            val_to_mask = None

        mask_code = code.detach().clone()
        # Sample the amount of tokens + localization to mask
        mask = torch.rand(size=code.size()) < val_to_mask.view(code.size(0), 1)

        if value > 0:  # Mask the selected token by the value
            mask_code[mask] = torch.full_like(mask_code[mask], value)
        else:  # Replace by a randon token
            mask_code[mask] = torch.randint_like(mask_code[mask], 0, codebook_size)

        return mask_code, mask

    def adap_sche(self, step, num_tokens = None, mode="arccos", leave=False):
        """ Create a sampling scheduler
           :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
           :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        r = torch.linspace(1, 0, step)
        if mode == "root":              # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":          # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":          # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":          # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":          # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            return

        # fill the scheduler by the ratio of tokens to predict at each step
        if num_tokens is None:
            num_tokens = self.num_tokens
        sche = (val_to_mask / val_to_mask.sum()) * (num_tokens)
        sche = sche.round()
        sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
        sche[-1] += (num_tokens) - sche.sum()         # need to sum up nb of code
        return tqdm(sche.int(), leave=leave)

    def train_one_epoch(self, log_iter=5000):
        """ Train the model for 1 epoch """
        self.vit.train()
        cum_loss = 0.
        window_loss = deque(maxlen=self.args.grad_cum)
        window_loss_img = deque(maxlen=self.args.grad_cum)  # for logging purpose only
        window_loss_lidar = deque(maxlen=self.args.grad_cum)# for logging purpose only
        bar = tqdm(self.train_data, leave=False) if self.args.is_master else self.train_data
        n = len(self.train_data)
        # Start training for 1 epoch
        # for x, y in bar:
        self.optim.zero_grad()

        for tok, x, lidar_path in bar: #tok_img, tok_lidar in bar:
            # x = x.to(self.args.device)
            # y = y.to(self.args.device)
            # x = 2 * x - 1  # normalize from x in [0,1] to [-1,1] for VQGAN

            # Drop xx% of the condition for cfg
            # drop_label = torch.empty(y.size()).uniform_(0, 1) < self.args.drop_label

            # VQGAN encoding to img tokens
            # with torch.no_grad():
            #     emb, _, [_, _, code] = self.ae.encode(x)
            #     code = code.reshape(x.size(0), self.patch_size, self.patch_size)

            #tok_img = tok_img.to(self.device)
            #tok_lidar = tok_img.to(self.device)
            #code = torch.cat(tok_img, tok_lidar)
            if self.img_only:
                tok = tok[:,:self.num_tokens_img]
            code = tok.to(self.args.device)
            x = x.to(self.args.device)
            x = 2 * x - 1
            
            # Mask the encoded tokens
            masked_code_img, mask_img = self.get_mask_code(code[:,:self.num_tokens_img], value=self.codebook_size_img, codebook_size=self.codebook_size_img)
            if self.img_only:
                masked_code = masked_code_img
                mask = mask_img
            else:
                masked_code_lidar, mask_lidar = self.get_mask_code(code[:,self.num_tokens_img:], value=self.codebook_size_lidar, codebook_size=self.codebook_size_lidar)
                masked_code = torch.cat((masked_code_img,masked_code_lidar),dim=1)
                mask = torch.cat((mask_img,mask_lidar),dim=1)

            with torch.cuda.amp.autocast():                             # half precision

                #print(f"masked_code {masked_code}")
                #print(f"masked_code shape {masked_code.shape}")
                pred_img, pred_lidar = self.vit(masked_code)  # The unmasked tokens prediction
                #print(f"pred {pred}")
                #print(f"pred shape {pred.shape}")

                # for logging purpose only
                #pred_img = pred[:,:self.num_tokens_img,:]
                #pred_lidar = pred[:,self.num_tokens_img:,:] if not self.img_only else None

                code_img = code[:,:self.num_tokens_img].contiguous()
                code_lidar = code[:,self.num_tokens_img:].contiguous() if not self.img_only else None

                loss_img = self.criterion(pred_img.reshape(-1, self.codebook_size_img), code_img.view(-1)) / self.args.grad_cum
                loss_lidar = self.criterion(pred_lidar.reshape(-1, self.codebook_size_lidar), code_lidar.view(-1)) / self.args.grad_cum if not self.img_only else torch.tensor([0.], device=self.args.device)

                # Cross-entropy loss
                if self.img_only:
                    loss =  loss_img
                else:
                    loss =  (loss_img + loss_lidar)/2

            # update weight if accumulation of gradient is done
            update_grad = self.args.iter % self.args.grad_cum == self.args.grad_cum - 1
            #if update_grad:
            #    self.optim.zero_grad()
            
            self.scaler.scale(loss).backward()  # rescale to get more precise loss

            if update_grad:
                self.scaler.unscale_(self.optim)                      # rescale loss
                nn.utils.clip_grad_norm_(self.vit.parameters(), 0.5)  # Clip gradient
                self.scaler.step(self.optim)
                self.scaler.update()
                if self.args.onecycle:
                    self.lr_scheduler.step()
                
                self.optim.zero_grad()

            cum_loss += loss.cpu().item()
            window_loss.append(loss.data.cpu().numpy().mean())
            window_loss_img.append(loss_img.data.cpu().numpy().mean())  # for logging purpose only
            window_loss_lidar.append(loss_lidar.data.cpu().numpy().mean())# for logging purpose only
            # logs
            if update_grad and self.args.is_master:
                lr = self.optim.param_groups[0]['lr']
                self.log_add_scalar('lr', lr, self.args.iter)
                self.log_add_scalar('Train/Loss', np.array(window_loss).sum(), self.args.iter)
                self.log_add_scalar('Train/LossImg', np.array(window_loss_img).sum(), self.args.iter)
                self.log_add_scalar('Train/LossLidar', np.array(window_loss_lidar).sum(), self.args.iter)

                print(f"\Loss {window_loss[-1]} LR {lr}")

            if self.args.iter % log_iter == 0 and self.args.is_master:
                # Load lidar and range project it
                if not self.img_only:
                    if self.mode == "range_image":
                        range_images = preprocess_range(lidar_path, self.args.lidar_config)
                        range_images = torch.from_numpy(range_images)
                        fake_mask = torch.ones(range_images.shape)  # trick...
                        range_images_to_pcd = torch.cat((range_images,fake_mask), dim=1)
                        _, input_pcds = self.lidar_tokenizer.process_output(range_images_to_pcd)
                        range_images = range_images.to(self.args.device)

                    elif self.mode == "voxel":
                        input_voxels = preprocess_voxel(lidar_path, self.args.lidar_config)
                        #voxels = voxels.to(self.args.device)
                        input_voxels[input_voxels <= 0.] = -99999   # next step is sigmoid...
                        input_pcds = self.lidar_tokenizer.process_output(input_voxels)

                # Generate sample for visualization
                gen_sample = self.sample(nb_sample=10)
                gen_sample_img = gen_sample[0]
                gen_sample_range_img = gen_sample[1]
                gen_sample_pcd = gen_sample[2] 

                gen_sample_img = vutils.make_grid(gen_sample_img, nrow=10, padding=2, normalize=True)
                self.log_add_img("Images/Sampling", gen_sample_img, self.args.iter)
                
                if not self.img_only:
                    if gen_sample_range_img is not None:
                        gen_sample_range_img = vutils.make_grid(gen_sample_range_img, nrow=10, padding=2, normalize=True)
                        self.log_add_img("Images/SamplingRange", gen_sample_range_img, self.args.iter)
                    gen_sample_pcd_images = torch.stack([pcd_to_plot_image(pcd) for pcd in gen_sample_pcd])
                    gen_sample_pcd_images = vutils.make_grid(gen_sample_pcd_images, nrow=10, padding=2, normalize=True)
                    self.log_add_img("Images/SamplingPcd", gen_sample_pcd_images, self.args.iter)

                # Show reconstruction
                unmasked_code_img = torch.softmax(pred_img, -1).max(-1)[1]
                unmasked_code_lidar = torch.softmax(pred_lidar, -1).max(-1)[1] if not self.img_only else None
                if self.img_only:
                    reco_sample_img, reco_sample_lidar, reco_sample_lidar_pcd = self.reco(x=x[:10], range_images=None, input_pcds=None, code=code[:10],
                                                                                          unmasked_code_img=unmasked_code_img[:10],
                                                                                          unmasked_code_lidar=None, mask=mask[:10])
                else:
                    if self.mode == "range_image":
                        reco_sample_img, reco_sample_lidar, reco_sample_lidar_pcd = self.reco(x=x[:10], range_images=range_images[:10], input_pcds=input_pcds[:10], code=code[:10],
                                                                                          unmasked_code_img=unmasked_code_img[:10],
                                                                                          unmasked_code_lidar=unmasked_code_lidar[:10], mask=mask[:10])
                    elif self.mode == "voxel":
                        reco_sample_img, reco_sample_lidar_pcd = self.reco(x=x[:10], range_images=None, input_pcds=input_pcds[:10], code=code[:10], unmasked_code_img=unmasked_code_img[:10],
                                                                                          unmasked_code_lidar=unmasked_code_lidar[:10], mask=mask[:10])

                reco_sample_img = vutils.make_grid(reco_sample_img.data, nrow=self.args.bsize, padding=2, normalize=True)
                self.log_add_img("Images/Reconstruction", reco_sample_img, self.args.iter)

                if not self.img_only:
                    all_lidar_imgs = []
                    for l in reco_sample_lidar_pcd:
                        for sample in l:
                            all_lidar_imgs.append(pcd_to_plot_image(sample))
                    all_lidar_imgs = torch.stack(all_lidar_imgs)

                    reco_sample_lidar_pcd = vutils.make_grid(all_lidar_imgs.data, nrow=self.args.bsize, padding=2, normalize=True)
                    self.log_add_img("Images/ReconstructionPcd", reco_sample_lidar_pcd, self.args.iter)
                    
                    if self.mode == "range_image":
                        reco_sample_lidar = vutils.make_grid(reco_sample_lidar.data, nrow=self.args.bsize, padding=2, normalize=True)
                        self.log_add_img("Images/ReconstructionRange", reco_sample_lidar, self.args.iter)

                # Save Network
                self.save_network(model=self.vit, path=self.args.vit_folder+"current.pth",
                                  iter=self.args.iter, optimizer=self.optim, lr_scheduler=self.lr_scheduler, global_epoch=self.args.global_epoch)

            self.args.iter += 1

        return cum_loss / n

    def fit(self):
        """ Train the model """
        if self.args.is_master:
            print("Start training:")

        start = time.time()
        # Start training
        for e in range(self.args.global_epoch, self.args.epoch):
            # synch every GPUs
            if self.args.is_multi_gpus:
                self.train_data.sampler.set_epoch(e)

            # Train for one epoch
            train_loss = self.train_one_epoch()

            # Synch loss
            if self.args.is_multi_gpus:
                train_loss = self.all_gather(train_loss, torch.cuda.device_count())

            # Save model
            if e % 5 == 0 and self.args.is_master:
                self.save_network(model=self.vit, path=self.args.vit_folder + f"epoch_{self.args.global_epoch:03d}.pth",
                                  iter=self.args.iter, optimizer=self.optim, lr_scheduler=self.lr_scheduler, global_epoch=self.args.global_epoch)

            # Clock time
            clock_time = (time.time() - start)
            if self.args.is_master:
                self.log_add_scalar('Train/GlobalLoss', train_loss, self.args.global_epoch)
                print(f"\rEpoch {self.args.global_epoch},"
                      f" Iter {self.args.iter :},"
                      f" Loss {train_loss:.4f},"
                      f" Time: {clock_time // 3600:.0f}h {(clock_time % 3600) // 60:.0f}min {clock_time % 60:.2f}s")
            self.args.global_epoch += 1
        
        if self.args.is_master:
            print("Saving final model:")
            self.save_network(model=self.vit, path=self.args.vit_folder+"last.pth",
                            iter=self.args.iter, optimizer=self.optim, lr_scheduler=self.lr_scheduler, global_epoch=self.args.global_epoch)
            print("Fin")
            
    def eval(self):
        """ Evaluation of the model"""
        self.vit.eval()
        if self.args.is_master:
            print(f"Evaluation with hyper-parameter ->\n"
                  f"scheduler: {self.args.sched_mode}, number of step: {self.args.step}, "
                  f"softmax temperature: {self.args.sm_temp}, cfg weight: {self.args.cfg_w}, "
                  f"gumbel temperature: {self.args.r_temp}")
        # Evaluate the model
        m = self.sae.compute_and_log_metrics(self)
        self.vit.train()
        return m
    
    def generate_samples(self):
        """ Evaluation of the model"""
        self.vit.eval()
        if self.args.is_master:
            print(f"Evaluation with hyper-parameter ->\n"
                  f"scheduler: {self.args.sched_mode}, number of step: {self.args.step}, "
                  f"softmax temperature: {self.args.sm_temp}, cfg weight: {self.args.cfg_w}, "
                  f"gumbel temperature: {self.args.r_temp}")
        # Evaluate the model
        print("Generate samples only")
        self.sae.sample_images(self)
        return
        #m = self.sae.compute_and_log_metrics(self)
        #self.vit.train()
        #return m

    def reco(self, x=None, range_images=None, input_pcds=None, code=None, masked_code=None, unmasked_code_img=None, unmasked_code_lidar=None, mask=None):
        """ For visualization, show the model ability to reconstruct masked img
           :param
            x             -> torch.FloatTensor: bsize x 3 x 256 x 256, the real image
            code          -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
            masked_code   -> torch.LongTensor: bsize x 16 x 16, the masked image tokens
            unmasked_code -> torch.LongTensor: bsize x 16 x 16, the prediction of the transformer
            mask          -> torch.LongTensor: bsize x 16 x 16, the binary mask of the encoded image
           :return
            l_visual      -> torch.LongTensor: bsize x 3 x (256 x ?) x 256, the visualization of the images
        """
        if code is not None:
            code_img = torch.clamp(code[:,:self.num_tokens_img], 0, self.codebook_size_img-1)
            #code_lidar = torch.clamp(code[:,self.num_tokens_img:], self.codebook_size_img, self.codebook_size_lidar-1) if not self.img_only else None
            code_lidar = torch.clamp(code[:,self.num_tokens_img:], 0, self.codebook_size_lidar-1) if not self.img_only else None

        if mask is not None:
            mask_img = mask[:,:self.num_tokens_img]
            mask_lidar = mask[:,self.num_tokens_img:] if not self.img_only else None

        if masked_code is not None:
            masked_code_img = torch.clamp(masked_code[:,:self.num_tokens_img], 0, self.codebook_size_img-1)
            masked_code_lidar = torch.clamp(masked_code[:,self.num_tokens_img:], 0, self.codebook_size_lidar-1) if not self.img_only else None

        if unmasked_code_img is not None:
            unmasked_code_img = torch.clamp(unmasked_code_img, 0, self.codebook_size_img-1)
        if unmasked_code_lidar is not None:
            unmasked_code_lidar = torch.clamp(unmasked_code_lidar, 0, self.codebook_size_lidar-1) if not self.img_only else None

        l_visual_img = [x]
        l_visual_lidar = [range_images] if not range_images is None else []
        l_visual_lidar_pcd = [input_pcds] if not self.img_only else []
        with torch.no_grad():
            if code is not None:
                # IMAGE
                code_img = code_img.view(code_img.size(0), self.patch_size, self.patch_size)
                # Decoding reel code
                _x = self.ae.decode_code(code_img)
                if mask is not None:
                    # Decoding reel code with mask to hide
                    mask_img = mask_img.view(code.size(0), 1, self.patch_size, self.patch_size).float()
                    __x2 = _x * (1 - F.interpolate(mask_img, (self.args.img_size, self.args.img_size)).to(self.args.device))
                    l_visual_img.append(__x2)
                
                # LIDAR
                if not self.img_only:
                    if self.mode=="range_image":
                        code_lidar = code_lidar.reshape(code_lidar.shape[0],self.args.num_tokens_lidar_h,self.args.num_tokens_lidar_w)
                        _range_image_rec = self.lidar_tokenizer.decode_code(code_lidar.contiguous())
                        _range_image_rec, pcd_recs = self.lidar_tokenizer.process_output(_range_image_rec)
                        _range_image_rec = _range_image_rec.to(self.args.device)

                        if mask is not None:
                            # Decoding reel code with mask to hide
                            mask_lidar = mask_lidar.view(code_lidar.shape[0], 1, self.args.num_tokens_lidar_h, self.args.num_tokens_lidar_w).float()
                            __range_image_rec2 = _range_image_rec * (1 - F.interpolate(mask_lidar, (_range_image_rec.shape[2], _range_image_rec.shape[3])).to(self.args.device))
                            l_visual_lidar.append(__range_image_rec2)
                            l_visual_lidar_pcd.append(pcd_recs)

                    elif self.mode=="voxel":
                        pcd_recs = self.lidar_tokenizer.decode_code(code_lidar.contiguous())
                        pcd_recs = self.lidar_tokenizer.process_output(pcd_recs)
                        l_visual_lidar_pcd.append(pcd_recs)

            if masked_code is not None:
                # IMAGE
                # Decoding masked code
                masked_code_img = masked_code_img.view(masked_code_img.size(0), self.patch_size, self.patch_size)
                __x = self.ae.decode_code(masked_code_img)
                l_visual_img.append(__x)

                # LIDAR
                if not self.img_only:
                    if self.mode=="range_image":
                        masked_code_lidar = masked_code_lidar.reshape(masked_code_lidar.shape[0],self.args.num_tokens_lidar_h,self.args.num_tokens_lidar_w)
                        masked_code_lidar = torch.clamp(masked_code_lidar, 0, self.codebook_size_lidar-1)
                        __range_image_rec = self.lidar_tokenizer.decode_code(masked_code_lidar.contiguous())
                        __range_image_rec, pcd_recs = self.lidar_tokenizer.process_output(__range_image_rec)
                        __range_image_rec = __range_image_rec.to(self.args.device)
                        l_visual_lidar.append(__range_image_rec)
                        l_visual_lidar_pcd.append(pcd_recs)
                    elif self.mode=="voxel":
                        pcd_recs = self.lidar_tokenizer.decode_code(masked_code_lidar.contiguous())
                        pcd_recs = self.lidar_tokenizer.process_output(pcd_recs)
                        l_visual_lidar_pcd.append(pcd_recs)

            if unmasked_code_img is not None:
                # IMAGE
                # Decoding predicted code
                unmasked_code_img = unmasked_code_img.view(unmasked_code_img.size(0), self.patch_size, self.patch_size)
                ___x = self.ae.decode_code(unmasked_code_img)
                l_visual_img.append(___x)

                # LIDAR
                if not self.img_only and unmasked_code_lidar is not None:
                    if self.mode=="range_image":
                        unmasked_code_lidar = unmasked_code_lidar.reshape(unmasked_code_lidar.shape[0],self.args.num_tokens_lidar_h,self.args.num_tokens_lidar_w)
                        unmasked_code_lidar = torch.clamp(unmasked_code_lidar, 0, self.codebook_size_lidar-1)
                        ___range_image_rec = self.lidar_tokenizer.decode_code(unmasked_code_lidar.contiguous())
                        ___range_image_rec, pcd_recs = self.lidar_tokenizer.process_output(___range_image_rec)
                        ___range_image_rec = ___range_image_rec.to(self.args.device)
                        l_visual_lidar.append(___range_image_rec)
                        l_visual_lidar_pcd.append(pcd_recs)
                    elif self.mode=="voxel":
                        pcd_recs = self.lidar_tokenizer.decode_code(unmasked_code_lidar.contiguous())
                        pcd_recs = self.lidar_tokenizer.process_output(pcd_recs)
                        l_visual_lidar_pcd.append(pcd_recs)

            l_visual_img = torch.cat(l_visual_img, dim=0)
            if self.mode=="range_image":
                if not self.img_only:
                    l_visual_lidar = torch.cat(l_visual_lidar, dim=0)
                return l_visual_img, l_visual_lidar, l_visual_lidar_pcd
            
            elif self.mode=="voxel":
                return l_visual_img, l_visual_lidar_pcd

    def sample(self, init_code=None, nb_sample=50, labels=None, sm_temp=1, w=3,
               randomize="linear", r_temp=4.5, sched_mode="arccos", step=12):
        """ Generate sample with the MaskGIT model
           :param
            init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            nb_sample   -> int:              the number of image to generated
            labels      -> torch.LongTensor: the list of classes to generate
            sm_temp     -> float:            the temperature before softmax
            w           -> float:            scale for the classifier free guidance
            randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp      -> float:            temperature for the randomness
            sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            step:       -> int:              number of step for the decoding
           :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        self.vit.eval()
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks
        with torch.no_grad():
            if init_code is not None:  # Start with a pre-define code
                code = init_code
                mask = (init_code == self.codebook_size).float().view(nb_sample, self.num_tokens)
            else:  # Initialize a code
                if self.img_only:
                    code = torch.full((nb_sample, self.num_tokens_img), self.codebook_size_img).to(self.args.device)
                    mask = torch.ones(nb_sample, self.num_tokens).to(self.args.device)
                else:
                    code_img = torch.full((nb_sample, self.num_tokens_img), self.codebook_size_img).to(self.args.device)
                    code_lidar = torch.full((nb_sample, self.num_tokens_lidar), self.codebook_size_lidar).to(self.args.device)
                    code = torch.cat((code_img,code_lidar), dim=1)
                    mask = torch.ones(nb_sample, self.num_tokens).to(self.args.device)

            # Instantiate scheduler
            if isinstance(sched_mode, str):  # Standard ones
                scheduler = self.adap_sche(step, mode=sched_mode)
            else:  # Custom one
                scheduler = sched_mode

            # Beginning of sampling, t = number of token to predict a step "indice"
            for indice, t in enumerate(scheduler):
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                with torch.cuda.amp.autocast():  # half precision
                    # TODO what is this classifier free guidance?
                    #if w != 0:
                        # Model Prediction
                        #logit = self.vit(code.clone())
                        #logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                        #_w = w * (indice / (len(scheduler)-1))
                        # Classifier Free Guidance
                        #logit = (1 + _w) * logit_c - _w * logit_u
                    #else:
                    logit_img, logit_lidar = self.vit(code.clone())

                prob_img = torch.softmax(logit_img * sm_temp, -1, dtype=torch.double)
                prob_lidar = torch.softmax(logit_lidar * sm_temp, -1, dtype=torch.double) if not self.img_only else None

                # Sample the code from the softmax prediction
                
                distri_img = torch.distributions.Categorical(probs=prob_img)
                distri_lidar = torch.distributions.Categorical(probs=prob_lidar) if not self.img_only else None

                pred_code_img = distri_img.sample()
                pred_code_lidar = distri_lidar.sample() if not self.img_only else None
                #print(f"pred_code_img {pred_code_img.shape}")
                #print(f"pred_code_lidar {pred_code_lidar.shape}")
                if self.img_only:
                    pred_code = pred_code_img
                    conf = torch.gather(prob_img, 2, pred_code.view(nb_sample, self.num_tokens, 1))
                else:
                    pred_code = torch.cat((pred_code_img, pred_code_lidar),dim=1)
                    conf_img = torch.gather(prob_img, 2, pred_code_img.view(nb_sample, self.num_tokens_img, 1))
                    conf_lidar = torch.gather(prob_lidar, 2, pred_code_lidar.view(nb_sample, self.num_tokens_lidar, 1))
                    conf = torch.cat((conf_img, conf_lidar),dim=1)


                if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                    ratio = (indice / (len(scheduler)-1))
                    #rand_img = r_temp * np.random.gumbel(size=(nb_sample, self.num_tokens_img)) * (1 - ratio)
                    #rand_lidar = r_temp * np.random.gumbel(size=(nb_sample, self.num_tokens_lidar)) * (1 - ratio)
                    #conf_img = torch.log(conf_img.squeeze()) + torch.from_numpy(rand_img).to(self.args.device)
                    #conf_lidar = torch.log(conf_lidar.squeeze()) + torch.from_numpy(rand_lidar).to(self.args.device)
                    rand = r_temp * np.random.gumbel(size=(nb_sample, self.num_tokens)) * (1 - ratio)
                    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.args.device)

                elif randomize == "warm_up":  # chose random sample for the 2 first steps
                    #conf_img = torch.rand_like(conf_img) if indice < 2 else conf_img
                    #conf_lidar = torch.rand_like(conf_lidar) if indice < 2 else conf_lidar
                    conf = torch.rand_like(conf) if indice < 2 else conf

                elif randomize == "random":   # chose random prediction at each step
                    #conf_img = torch.rand_like(conf_img)
                    #conf_lidar = torch.rand_like(conf_lidar)
                    conf = torch.rand_like(conf)

                # do not predict on already predicted tokens
                #conf_img[~mask_img.bool()] = -math.inf
                #conf_lidar[~mask_lidar.bool()] = -math.inf
                conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                conf = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.num_tokens)
                f_mask = (mask.view(nb_sample, self.num_tokens).float() * conf.view(nb_sample, self.num_tokens).float()).bool()
                code[f_mask] = pred_code.view(nb_sample, self.num_tokens)[f_mask]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                l_codes.append(pred_code.view(nb_sample, self.num_tokens).clone())
                l_mask.append(mask.view(nb_sample, self.num_tokens).clone())

            # decode the final prediction

            #_code = torch.clamp(code, 0,  self.codebook_size-1)
            # separate image and lidar tokens
            img_code = code[:,:self.num_tokens_img]
            lidar_code = code[:,self.num_tokens_img:] if not self.img_only else None

            # decode image
            #print(f"num image tokens over: {(img_code >= self.codebook_size_img).sum()}")
            #print(f"image tokens over: {img_code[img_code >= self.codebook_size_img]}")
            img_code = img_code.reshape(img_code.shape[0],self.patch_size,self.patch_size)
            img_code = torch.clamp(img_code, 0, self.codebook_size_img-1)
            x = self.ae.decode_code(img_code.contiguous())

            # decode lidar
            range_images_rec = None
            pcd_recs = None

            if not self.img_only:
                if self.mode == "range_image":
                    lidar_code = lidar_code.reshape(lidar_code.shape[0],self.args.num_tokens_lidar_h,self.args.num_tokens_lidar_w)
                    lidar_code = torch.clamp(lidar_code, 0, self.codebook_size_lidar-1)
                    range_image_rec = self.lidar_tokenizer.decode_code(lidar_code.contiguous())
                    range_images_rec, pcd_recs = self.lidar_tokenizer.process_output(range_image_rec)
                elif self.mode == "voxel":
                    lidar_code = torch.clamp(lidar_code, 0, self.codebook_size_lidar-1)
                    pcd_recs = self.lidar_tokenizer.decode_code(lidar_code.contiguous())
                    pcd_recs = self.lidar_tokenizer.process_output(pcd_recs)

        self.vit.train()
        return x, range_images_rec, pcd_recs, l_codes, l_mask


    def conditioned(self, sm_temp=1, max_images=100,
               randomize="linear", r_temp=4.5, sched_mode="arccos", step=12, img2pcd=False, remask=False, k=100):
        
        from matplotlib.colors import ListedColormap

        cmap = plt.cm.coolwarm  # Start with the 'coolwarm' colormap
        new_cmap = cmap(np.linspace(0, 1, cmap.N))  # Get the color range
        new_cmap[0, :] = [0, 0, 0, 1]  # Set the lowest color to black (RGBA format)
        custom_cmap = ListedColormap(new_cmap)
        
        #k = 10#int(self.num_tokens_img*0.9)
        self.vit.eval()
        c = 0

        init_mask_both = False
        
        with torch.no_grad():

            bar = tqdm(self.test_data)
            for i, (tok, img, lidar_path) in enumerate(bar):
                
                tok = tok.to(self.args.device)
                img = img.to(self.args.device)

                img_tokens = tok[:,:self.num_tokens_img]
                lidar_tokens = tok[:,self.num_tokens_img:]

                bs = img_tokens.shape[0]

                c += bs
                if c >= max_images:
                    return

                if init_mask_both:
                    code_img = img_tokens.clone() #torch.full(img_tokens.shape, self.codebook_size_img).to(self.args.device)
                    code_lidar = lidar_tokens.clone()

                    # INITIAL MASKING
                    inds_to_mask_lidar = random.sample(list(range(self.num_tokens_lidar)), k)  ###
                    code_lidar[:,inds_to_mask_lidar] = self.codebook_size_lidar  ###

                    inds_to_mask_img = random.sample(list(range(self.num_tokens_img)), k)  ###
                    code_img[:,inds_to_mask_img] = self.codebook_size_img  ###
                    code = torch.cat((code_img,code_lidar), dim=1)
                    mask = torch.zeros(bs, self.num_tokens).to(self.args.device)
                    mask[:,inds_to_mask_img] = 1
                
                elif img2pcd:
                    code_img = img_tokens.clone()
                    code_lidar = torch.full(lidar_tokens.shape, self.codebook_size_lidar).to(self.args.device)

                    # INITIAL MASKING
                    inds_to_mask_img = random.sample(list(range(self.num_tokens_img)), k)
                    code_img[:,inds_to_mask_img] = self.codebook_size_img

                    code = torch.cat((code_img,code_lidar), dim=1)
                    mask = torch.zeros(bs, self.num_tokens).to(self.args.device)
                    mask[:,self.num_tokens_img:] = 1

                else:
                    code_img = torch.full(img_tokens.shape, self.codebook_size_img).to(self.args.device)
                    code_lidar = lidar_tokens.clone()

                    # INITIAL MASKING
                    inds_to_mask_lidar = random.sample(list(range(self.num_tokens_lidar)), k)  ###
                    code_lidar[:,inds_to_mask_lidar] = self.codebook_size_lidar  ###

                    code = torch.cat((code_img,code_lidar), dim=1)
                    mask = torch.zeros(bs, self.num_tokens).to(self.args.device)
                    mask[:,:self.num_tokens_img] = 1

                init_mask = mask.clone()
                """ code_img = torch.full((bs, self.num_tokens_img), self.codebook_size_img).to(self.args.device)
                code_lidar = torch.full((bs, self.num_tokens_lidar), self.codebook_size_lidar).to(self.args.device)
                code = torch.cat((code_img,code_lidar), dim=1)
                mask = torch.ones(bs, self.num_tokens).to(self.args.device) """

                # Instantiate scheduler
                num_tokens_masked = int(mask.sum()/bs)

                if isinstance(sched_mode, str):  # Standard ones
                    scheduler = self.adap_sche(step, mode=sched_mode, num_tokens=num_tokens_masked)# + int(self.num_tokens_lidar/200))    ###
                else:  # Custom one
                    scheduler = sched_mode

                # Beginning of sampling, t = number of token to predict a step "indice"
                for indice, t in enumerate(scheduler):
                    if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                        t = int(mask.sum().item())

                    if mask.sum() == 0:  # Break if code is fully predicted
                        break
                        
                    # RANDOMLY MASK k TOKENS EACH ITERATION
                    if remask:
                        if img2pcd:
                            inds_to_mask = np.array(random.sample(list(range(self.num_tokens_img)), k))  ###
                            code[:,:self.num_tokens_img] = img_tokens
                            code[:,inds_to_mask] = self.codebook_size_img  ###

                        else:
                            inds_to_mask = np.array(random.sample(list(range(self.num_tokens_lidar)), k))  ###
                            code[:,self.num_tokens_img:] = lidar_tokens
                            code[:,self.num_tokens_img + inds_to_mask] = self.codebook_size_lidar  ###

                    with torch.cuda.amp.autocast():  # half precision
                        logit_img, logit_lidar = self.vit(code.clone())

                    prob_img = torch.softmax(logit_img * sm_temp, -1, dtype=torch.double)
                    prob_lidar = torch.softmax(logit_lidar * sm_temp, -1, dtype=torch.double) if not self.img_only else None

                    # Sample the code from the softmax prediction
                    
                    distri_img = torch.distributions.Categorical(probs=prob_img)
                    distri_lidar = torch.distributions.Categorical(probs=prob_lidar) if not self.img_only else None

                    pred_code_img = distri_img.sample()
                    pred_code_lidar = distri_lidar.sample() if not self.img_only else None

                    if self.img_only:
                        pred_code = pred_code_img
                        conf = torch.gather(prob_img, 2, pred_code.view(bs, self.num_tokens, 1))
                    else:
                        pred_code = torch.cat((pred_code_img, pred_code_lidar),dim=1)
                        conf_img = torch.gather(prob_img, 2, pred_code_img.view(bs, self.num_tokens_img, 1))
                        conf_lidar = torch.gather(prob_lidar, 2, pred_code_lidar.view(bs, self.num_tokens_lidar, 1))
                        conf = torch.cat((conf_img, conf_lidar),dim=1)

                    if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                        ratio = (indice / (len(scheduler)-1))
                        #rand_img = r_temp * np.random.gumbel(size=(nb_sample, self.num_tokens_img)) * (1 - ratio)
                        #rand_lidar = r_temp * np.random.gumbel(size=(nb_sample, self.num_tokens_lidar)) * (1 - ratio)
                        #conf_img = torch.log(conf_img.squeeze()) + torch.from_numpy(rand_img).to(self.args.device)
                        #conf_lidar = torch.log(conf_lidar.squeeze()) + torch.from_numpy(rand_lidar).to(self.args.device)
                        rand = r_temp * np.random.gumbel(size=(bs, self.num_tokens)) * (1 - ratio)
                        conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.args.device)

                    elif randomize == "warm_up":  # chose random sample for the 2 first steps
                        #conf_img = torch.rand_like(conf_img) if indice < 2 else conf_img
                        #conf_lidar = torch.rand_like(conf_lidar) if indice < 2 else conf_lidar
                        conf = torch.rand_like(conf) if indice < 2 else conf

                    elif randomize == "random":   # chose random prediction at each step
                        #conf_img = torch.rand_like(conf_img)
                        #conf_lidar = torch.rand_like(conf_lidar)
                        conf = torch.rand_like(conf)

                    conf[~mask.bool()] = -math.inf

                    # chose the predicted token with the highest confidence
                    tresh_conf, indice_mask = torch.topk(conf.view(bs, -1), k=t, dim=-1)
                    tresh_conf = tresh_conf[:, -1]

                    # replace the chosen tokens
                    conf = (conf >= tresh_conf.unsqueeze(-1)).view(bs, self.num_tokens)
                    f_mask = (mask.view(bs, self.num_tokens).float() * conf.view(bs, self.num_tokens).float()).bool()
                    code[f_mask] = pred_code.view(bs, self.num_tokens)[f_mask]

                    # update the mask
                    for i_mask, ind_mask in enumerate(indice_mask):
                        mask[i_mask, ind_mask] = 0
                    
                # separate image and lidar tokens
                img_code = code[:,:self.num_tokens_img]
                lidar_code = code[:,self.num_tokens_img:]

                # decode image
                img_code = img_code.reshape(img_code.shape[0],self.patch_size,self.patch_size)
                img_code = torch.clamp(img_code, 0, self.codebook_size_img-1)
                x = self.ae.decode_code(img_code.contiguous())

                x = x.float()
                x = (x + 1)/2
                x = torch.clamp(x, 0., 1.)
                
                # masked image vis
                mask_img = init_mask[:,:self.num_tokens_img]
                mask_img = mask_img.view(code.size(0), 1, self.patch_size, self.patch_size).float()
                x_masked = x * (1 - F.interpolate(mask_img, (self.args.img_size, self.args.img_size)).to(self.args.device))
                x_masked = x_masked.cpu()   
                
                x = x.cpu()
                img = img.cpu()

                # decode lidar
                range_images_rec = None
                if self.mode == "range_image":
                    # MODEL OUTPUT
                    lidar_code = lidar_code.reshape(lidar_code.shape[0],self.args.num_tokens_lidar_h,self.args.num_tokens_lidar_w)
                    lidar_code = torch.clamp(lidar_code, 0, self.codebook_size_lidar-1)
                    range_image_rec = self.lidar_tokenizer.decode_code(lidar_code.contiguous())
                    range_images_rec, pcd_recs = self.lidar_tokenizer.process_output(range_image_rec)

                    range_images_rec = range_images_rec*0.5 + 0.5
                    range_images_rec = range_images_rec*self.lidar_config['model']['params']['lossconfig']['params']['dataset_config']['depth_scale']
                    if self.lidar_config['model']['params']['lossconfig']['params']['dataset_config']['log_scale']:
                        range_images_rec = np.exp2(range_images_rec) - 1
                    range_images_rec = range_images_rec.permute(0, 2, 3, 1)
                    range_images_rec = range_images_rec.numpy()

                    # GROUND TRUTH PCD
                    range_images = preprocess_range(lidar_path, self.args.lidar_config)
                    range_images = torch.from_numpy(range_images)
                    fake_mask = torch.ones(range_images.shape)  # trick...
                    range_images_to_pcd = torch.cat((range_images,fake_mask), dim=1)
                    _, input_pcds = self.lidar_tokenizer.process_output(range_images_to_pcd)
                    range_images = range_images.cpu()

                    range_images = range_images*0.5 + 0.5
                    range_images = range_images*self.lidar_config['model']['params']['lossconfig']['params']['dataset_config']['depth_scale']
                    if self.lidar_config['model']['params']['lossconfig']['params']['dataset_config']['log_scale']:
                        range_images = np.exp2(range_images) - 1
                    range_images = range_images.permute(0, 2, 3, 1)
                    range_images = range_images.numpy()
                    
                elif self.mode == "voxel":
                    # MODEL OUTPUT
                    lidar_code = torch.clamp(lidar_code, 0, self.codebook_size_lidar-1)
                    pcd_recs = self.lidar_tokenizer.decode_code(lidar_code.contiguous())
                    pcd_recs = self.lidar_tokenizer.process_output(pcd_recs)
                    
                    # GROUND TRUTH PCD
                    input_voxels = preprocess_voxel(lidar_path, self.args.lidar_config)
                    #voxels = voxels.to(self.args.device)
                    input_voxels[input_voxels <= 0.] = -99999   # next step is sigmoid...
                    input_pcds = self.lidar_tokenizer.process_output(input_voxels)



                if not range_images_rec is None:
                    fig, axes = plt.subplots(bs, 4, figsize=(10, 5 * bs))
                    for i in range(bs):
                            if bs > 1:
                                ax1, ax2, ax3, ax4 = axes[i]
                            else:
                                ax1, ax2, ax3, ax4 = axes
                            ax1.imshow(img[i].permute(1, 2, 0).numpy())
                            ax2.imshow(range_images[i], cmap=custom_cmap, vmin=0, vmax=160)
                            ax3.imshow(x[i].permute(1, 2, 0).numpy())
                            ax4.imshow(range_images_rec[i], cmap=custom_cmap, vmin=0, vmax=160)
                            ax1.axis('off')
                            ax2.axis('off')
                            ax3.axis('off')
                            ax4.axis('off')
                else:
                    fig, axes = plt.subplots(bs, 3, figsize=(10, 5 * bs))
                    for i in range(bs):
                            ax1, ax2, ax3 = axes[i]
                            ax3.imshow(x[i].permute(1, 2, 0).numpy())
                            ax1.imshow(img[i].permute(1, 2, 0).numpy())
                            ax2.imshow(x_masked[i].permute(1, 2, 0).numpy())
                            ax3.axis('off')
                            ax1.axis('off')
                            ax2.axis('off')

                plt.tight_layout()
                if img2pcd:
                    plt.savefig(f'saved_img/img2pcd_{c}_{k}.png')
                else:
                    plt.savefig(f'saved_img/pcd2img_{c}_{k}.png')
                plt.close()

        return x

    def plot_images(self, save_path="saved_img/image_only_samples.png"):
        gen_sample = self.sample(nb_sample=16,
                                sm_temp=self.args.sm_temp,
                                w=self.args.cfg_w,
                                randomize="linear",
                                r_temp=self.args.r_temp,
                                sched_mode=self.args.sched_mode,
                                step=self.args.step)
        imgs = gen_sample[0].cpu()
        print(imgs.shape)
        imgs = (imgs+1.)/2.
        imgs = torch.clamp(imgs, 0, 1)
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = imgs.numpy()
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))

        for i, ax in enumerate(axes.flatten()):
            ax.imshow(imgs[i])
            ax.axis('off') 

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def plot_img_range_img(self, save_path="saved_img/image_range_image.png"):
        from matplotlib.colors import ListedColormap

        cmap = plt.cm.coolwarm  # Start with the 'coolwarm' colormap
        new_cmap = cmap(np.linspace(0, 1, cmap.N))  # Get the color range
        new_cmap[0, :] = [0, 0, 0, 1]  # Set the lowest color to black (RGBA format)
        custom_cmap = ListedColormap(new_cmap)

        gen_sample = self.sample(nb_sample=8,
                                sm_temp=self.args.sm_temp,
                                w=self.args.cfg_w,
                                randomize="linear",
                                r_temp=self.args.r_temp,
                                sched_mode=self.args.sched_mode,
                                step=self.args.step)
        imgs = gen_sample[0].cpu()
        range_imgs = gen_sample[1].cpu()
        imgs = (imgs+1.)/2.
        imgs = torch.clamp(imgs, 0, 1)
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = imgs.numpy()

        range_imgs = range_imgs*0.5 + 0.5
        range_imgs = range_imgs*self.lidar_config['model']['params']['lossconfig']['params']['dataset_config']['depth_scale']
        if self.lidar_config['model']['params']['lossconfig']['params']['dataset_config']['log_scale']:
            range_imgs = np.exp2(range_imgs) - 1
        range_imgs = range_imgs.permute(0, 2, 3, 1)
        range_imgs = range_imgs.numpy()
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))

        for i, ax in enumerate(axes.flatten()):
            if i % 2 == 0:
                ax.imshow(imgs[int(i//2)])
                ax.axis('off') 
            else:
                im = ax.imshow(range_imgs[int(i//2)], cmap=custom_cmap, vmin=0, vmax=160)
                ax.axis('off') 

        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Adjust position and size as needed
        fig.colorbar(im, cax=cbar_ax)

        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the colorbar
        plt.savefig(save_path)
        plt.close()


    def plot_pcd(self, save_folder="saved_img"):

        gen_sample = self.sample(nb_sample=10,
                                sm_temp=self.args.sm_temp,
                                w=self.args.cfg_w,
                                randomize="linear",
                                r_temp=self.args.r_temp,
                                sched_mode=self.args.sched_mode,
                                step=self.args.step)

        for i in range(len(gen_sample[2])):
            pcds = gen_sample[2][i].cpu()
            pcds = pcds.numpy()

            np.savez(f"{save_folder}/pcd_{i}",pcds)

            imgs = gen_sample[0][i].cpu()
            imgs = (imgs+1.)/2.
            imgs = torch.clamp(imgs, 0, 1)
            imgs = imgs.permute(1, 2, 0)
            imgs = imgs.numpy()
            
            fig, ax = plt.subplots(1, figsize=(5, 5))

            ax.imshow(imgs)
            ax.axis('off') 

            plt.tight_layout()
            plt.savefig(f"{save_folder}/pcd_img_{i}.png")
            plt.close()