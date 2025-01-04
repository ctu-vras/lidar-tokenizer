import torch
from torch import nn

try:
    from model_managment.helpers import pad_tensor
    from model_managment.losses.helpers import GeoConverter, weights_init, l1, l2, hinge_d_loss, vanilla_d_loss, measure_perplexity, square_dist_loss
    from model_managment.losses.discriminator import NLayerDiscriminator, LiDARNLayerDiscriminator, LiDARNLayerDiscriminatorV2
    from model_managment.losses.perceptual import PerceptualLoss
except Exception as e:
    print(e)
    print("\nTRYING WITH RELATIVE IMPORTS, BUT IF THE ISSUE WAS SOMETHING ELSE THEN YOU ARE PROBABLY MISSING A PACKAGE\n")
    from ...model_managment.helpers import pad_tensor
    from ...model_managment.losses.helpers import GeoConverter, weights_init, l1, l2, hinge_d_loss, vanilla_d_loss, measure_perplexity, square_dist_loss
    from ...model_managment.losses.discriminator import NLayerDiscriminator, LiDARNLayerDiscriminator, LiDARNLayerDiscriminatorV2
    from ...model_managment.losses.perceptual import PerceptualLoss

VERSION2DISC = {'v0': NLayerDiscriminator, 'v1': LiDARNLayerDiscriminator, 'v2': LiDARNLayerDiscriminatorV2}

from torchmetrics.regression import MeanSquaredError,MeanAbsoluteError

""" Adapted from https://github.com/hancyran/LiDAR-Diffusion """
class LidarDiffusionLossNoGan(nn.Module):
    # original name VQGeoLPIPSWithDiscriminator
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_out_channels=1, disc_factor=1.0, disc_weight=1.0,
                 mask_factor=0.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", n_classes=None, pixel_loss="l1", disc_version='v1',
                 geo_factor=1.0, curve_length=4, perceptual_factor=1.0, perceptual_type='rangenet_final',
                 root_dir="/home/LiDAR_Tokenizer/",
                 dataset_config=dict()):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert pixel_loss in ["l1", "l2"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.mask_factor = mask_factor
        self.geo_factor = geo_factor

        # scale of reconstruction loss
        self.rec_scale = 1
        if mask_factor > 0:
            self.rec_scale += 1.
        if geo_factor > 0:
            self.rec_scale += 1.
        if perceptual_factor > 0:
            self.rec_scale += 1.

        if pixel_loss == "l1":
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2

        self.perceptual_factor = perceptual_factor
        if perceptual_factor > 0.:
            print(f"{self.__class__.__name__}: Running with LPIPS.")
            self.perceptual_loss = PerceptualLoss(perceptual_type, dataset_config.depth_scale,
                                                  dataset_config.log_scale, root_dir=root_dir).eval()

        self.n_classes = n_classes

        self.geometry_converter = GeoConverter(curve_length, False, dataset_config)  # force converting xyz output
        self.geo_loss = square_dist_loss
        self.depth_range = dataset_config.depth_range

        self.train_img_mse = MeanSquaredError()
        self.train_img_mae = MeanAbsoluteError()
        self.train_points_mse = MeanSquaredError()
        self.train_points_mae = MeanAbsoluteError()
        self.train_mask_mse = MeanSquaredError()
        self.train_mask_mae = MeanAbsoluteError()
        #self.train_points_clamped_mse = MeanSquaredError()
        #self.train_points_clamped_mae = MeanAbsoluteError()
        self.val_img_mse = MeanSquaredError()
        self.val_img_mae = MeanAbsoluteError()
        self.val_points_mse = MeanSquaredError()
        self.val_points_mae = MeanAbsoluteError()
        self.val_mask_mse = MeanSquaredError()
        self.val_mask_mae = MeanAbsoluteError()
        #self.val_points_clamped_mse = MeanSquaredError()
        #self.val_points_clamped_mae = MeanAbsoluteError()

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", predicted_indices=None, masks=None, mask_invalid_in_loss=False, ratios=None):
        
        if ratios is not None:
            h,w,ratios = ratios

            # Assume all h and w are the same in the batch
            if isinstance(h, torch.Tensor):
                h = h[0]
                w = w[0]

            inputs = pad_tensor(inputs, h, w, ratios)
            masks = pad_tensor(masks, h, w, ratios)
            reconstructions = pad_tensor(reconstructions, h, w, ratios)
            
        rec_x = reconstructions[:, 0:1]
        rec_mask = reconstructions[:, 1:2]

        input_coord = self.geometry_converter(inputs)  
        rec_coord = self.geometry_converter(rec_x)

        mask_inds = torch.where(masks==1)
        if mask_invalid_in_loss:
            inputs_masked = inputs[             mask_inds[0],mask_inds[1],mask_inds[2],mask_inds[3]]
            rec_x_masked = reconstructions[     mask_inds[0],mask_inds[1],mask_inds[2],mask_inds[3]]
            input_coord_masked = input_coord[   mask_inds[0],:,mask_inds[2],mask_inds[3]]
            rec_coord_masked = rec_coord[       mask_inds[0],:,mask_inds[2],mask_inds[3]]      
        else:
            inputs_masked = inputs
            rec_x_masked = reconstructions
            input_coord_masked = input_coord
            rec_coord_masked = rec_coord

        ############# Reconstruction #############
        # pixel reconstruction loss
        if self.mask_factor > 0 and masks is not None:
            pixel_rec_loss = self.pixel_loss(inputs_masked.contiguous(), rec_x_masked.contiguous())
            mask_rec_loss = self.pixel_loss(masks.contiguous(), rec_mask.contiguous()) * self.mask_factor
        else:
            pixel_rec_loss = self.pixel_loss(inputs_masked.contiguous(), rec_x_masked.contiguous())
            mask_rec_loss = torch.tensor(0.0)

        # geometry reconstruction loss (bev)
        if self.geo_factor > 0:
            geo_rec_loss = self.geo_loss(input_coord_masked[:, :2].contiguous(), rec_coord_masked[:, :2].contiguous()) * self.geo_factor
            geo_rec_loss = geo_rec_loss.squeeze()
        else:
            geo_rec_loss = torch.tensor(0.0)

        # perceptual loss
        if self.perceptual_factor > 0:
            perceptual_loss = self.perceptual_loss((inputs.contiguous(), input_coord.contiguous()),
                                                   (rec_x.contiguous(), rec_coord.contiguous())) * self.perceptual_factor
        else:
            perceptual_loss = torch.tensor(0.0)


        # overall reconstruction loss
        rec_loss = (pixel_rec_loss + torch.mean(mask_rec_loss) + geo_rec_loss + perceptual_loss) / self.rec_scale
        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

       
        # update generator (input: img, mask, coord, [cond])
        if optimizer_idx == 0:

            loss = nll_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/pix_rec_loss".format(split): pixel_rec_loss.detach().mean(),
                   "{}/geo_rec_loss".format(split): geo_rec_loss.detach().mean(),
                   "{}/mask_rec_loss".format(split): mask_rec_loss.detach().mean(),
                   "{}/perceptual_loss".format(split): perceptual_loss.detach().mean()}
            
            if split=="train":
                self.train_img_mse(rec_x_masked.contiguous(), inputs_masked.contiguous())
                self.train_img_mae(rec_x_masked.contiguous(), inputs_masked.contiguous())
                self.train_points_mse(rec_coord_masked, input_coord_masked)
                self.train_points_mae(rec_coord_masked, input_coord_masked)
                self.train_mask_mse(rec_mask, masks)
                self.train_mask_mae(rec_mask, masks)
                #self.train_points_clamped_mse(torch.clamp(rec_coord, self.depth_range[0], self.depth_range[1]), input_coord)
                #self.train_points_clamped_mae(torch.clamp(rec_coord, self.depth_range[0], self.depth_range[1]), input_coord)
                log[f"{split}/mask_mse"] = self.train_mask_mse
                log[f"{split}/mask_mae"] = self.train_mask_mae
                log[f"{split}/img_mse"] = self.train_img_mse
                log[f"{split}/img_mae"] = self.train_img_mae
                log[f"{split}/points_mse"] = self.train_points_mse
                log[f"{split}/points_mae"] = self.train_points_mae
                #log[f"{split}/points_clamped_mse"] = self.train_points_clamped_mse
                #log[f"{split}/points_clamped_mae"] = self.train_points_clamped_mae
            
            if split=="val":
                self.val_img_mse(rec_x_masked.contiguous(), inputs_masked.contiguous())
                self.val_img_mae(rec_x_masked.contiguous(), inputs_masked.contiguous()) 
                self.val_points_mse(rec_coord_masked, input_coord_masked)
                self.val_points_mae(rec_coord_masked, input_coord_masked)
                self.val_mask_mse(rec_mask, masks)
                self.val_mask_mae(rec_mask, masks)
                #self.val_points_clamped_mse(torch.clamp(rec_coord, self.depth_range[0], self.depth_range[1]), input_coord)
                #self.val_points_clamped_mae(torch.clamp(rec_coord, self.depth_range[0], self.depth_range[1]), input_coord)
                log[f"{split}/mask_mse"] = self.val_mask_mse
                log[f"{split}/mask_mae"] = self.val_mask_mae
                log[f"{split}/img_mse"] = self.val_img_mse
                log[f"{split}/img_mae"] = self.val_img_mae
                log[f"{split}/points_mse"] = self.val_points_mse
                log[f"{split}/points_mae"] = self.val_points_mae
                #log[f"{split}/points_clamped_mse"] = self.val_points_clamped_mse
                #log[f"{split}/points_clamped_mae"] = self.val_points_clamped_mae

            if predicted_indices is not None:
                assert self.n_classes is not None
                with torch.no_grad():
                    perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_classes)
                log[f"{split}/perplexity"] = perplexity
                log[f"{split}/cluster_usage"] = cluster_usage
            return loss, log
        

# Bellow is the loss from LD paper which includes the discriminator loss. Not used in the thesis but left here in any case.
""" 

class LidarDiffusionLoss(nn.Module):
    # original name VQGeoLPIPSWithDiscriminator
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_out_channels=1, disc_factor=1.0, disc_weight=1.0,
                 mask_factor=0.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", n_classes=None, pixel_loss="l1", disc_version='v1',
                 geo_factor=1.0, curve_length=4, perceptual_factor=1.0, perceptual_type='rangenet_final',
                 root_dir="/home/LiDAR_Tokenizer/",
                 dataset_config=dict()):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert pixel_loss in ["l1", "l2"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.mask_factor = mask_factor
        self.geo_factor = geo_factor

        # scale of reconstruction loss
        self.rec_scale = 1
        if mask_factor > 0:
            self.rec_scale += 1.
        if geo_factor > 0:
            self.rec_scale += 1.
        if perceptual_factor > 0:
            self.rec_scale += 1.

        if pixel_loss == "l1":
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2

        self.perceptual_factor = perceptual_factor
        if perceptual_factor > 0.:
            print(f"{self.__class__.__name__}: Running with LPIPS.")
            self.perceptual_loss = PerceptualLoss(perceptual_type, dataset_config.depth_scale,
                                                  dataset_config.log_scale, root_dir=root_dir).eval()

        disc_cls = VERSION2DISC[disc_version]
        self.discriminator = disc_cls(input_nc=disc_in_channels,
                                      output_nc=disc_out_channels,
                                      n_layers=disc_num_layers,
                                      use_actnorm=use_actnorm,
                                      ndf=disc_ndf).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQGeoLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.n_classes = n_classes

        self.geometry_converter = GeoConverter(curve_length, False, dataset_config)  # force converting xyz output
        self.geo_loss = square_dist_loss
        self.depth_range = dataset_config.depth_range

        self.train_img_mse = MeanSquaredError()
        self.train_img_mae = MeanAbsoluteError()
        self.train_points_mse = MeanSquaredError()
        self.train_points_mae = MeanAbsoluteError()
        #self.train_points_clamped_mse = MeanSquaredError()
        #self.train_points_clamped_mae = MeanAbsoluteError()
        self.val_img_mse = MeanSquaredError()
        self.val_img_mae = MeanAbsoluteError()
        self.val_points_mse = MeanSquaredError()
        self.val_points_mae = MeanAbsoluteError()
        #self.val_points_clamped_mse = MeanSquaredError()
        #self.val_points_clamped_mae = MeanAbsoluteError()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", predicted_indices=None, masks=None, mask_invalid_in_loss=False):
        
        if mask_invalid_in_loss:
            inputs = inputs[:,:,torch.where(masks==1)[2],torch.where(masks==1)[3]]
            rec_x = reconstructions[:,0:1,torch.where(masks==1)[2],torch.where(masks==1)[3]]
        else:
            rec_x = reconstructions[:, 0:1]
            
        input_coord = self.geometry_converter(inputs)
        rec_coord = self.geometry_converter(rec_x)

        ############# Reconstruction #############
        # pixel reconstruction loss
        if self.mask_factor > 0 and masks is not None:
            pixel_rec_loss = self.pixel_loss(inputs.contiguous(), rec_x.contiguous())
            mask_rec_loss = self.pixel_loss(masks.contiguous(), reconstructions[:, 1:2].contiguous()) * self.mask_factor
        else:
            pixel_rec_loss = self.pixel_loss(inputs.contiguous(), reconstructions.contiguous())
            mask_rec_loss = torch.tensor(0.0)

        # geometry reconstruction loss (bev)
        if self.geo_factor > 0:
            geo_rec_loss = self.geo_loss(input_coord[:, :2].contiguous(), rec_coord[:, :2].contiguous()) * self.geo_factor
        else:
            geo_rec_loss = torch.tensor(0.0)

        # perceptual loss
        if self.perceptual_factor > 0:
            perceptual_loss = self.perceptual_loss((inputs.contiguous(), input_coord.contiguous()),
                                                   (rec_x.contiguous(), rec_coord.contiguous())) * self.perceptual_factor
        else:
            perceptual_loss = torch.tensor(0.0)

        # overall reconstruction loss
        rec_loss = (pixel_rec_loss + mask_rec_loss + geo_rec_loss + perceptual_loss) / self.rec_scale
        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        ############# GAN #############
        disc_factor = 0. if global_step < self.discriminator_iter_start else self.disc_factor   #BUG ??
        # update generator (input: img, mask, coord, [cond])
        if optimizer_idx == 0:
            disc_recons = reconstructions.contiguous()
            if self.geo_factor > 0:
                disc_recons = torch.cat((disc_recons, rec_coord[:, :2].contiguous()), dim=1)
            if cond is not None and self.disc_conditional:
                disc_recons = torch.cat((disc_recons, cond), dim=1)
            logits_fake = self.discriminator(disc_recons)

            # adversarial loss
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/pix_rec_loss".format(split): pixel_rec_loss.detach().mean(),
                   "{}/geo_rec_loss".format(split): geo_rec_loss.detach().mean(),
                   "{}/mask_rec_loss".format(split): mask_rec_loss.detach().mean(),
                   "{}/perceptual_loss".format(split): perceptual_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean()}
            
            if split=="train":
                self.train_img_mse(reconstructions[:, 0:1].contiguous(), inputs.contiguous())
                self.train_img_mae(reconstructions[:, 0:1].contiguous(), inputs.contiguous()) 
                self.train_points_mse(rec_coord, input_coord)
                self.train_points_mae(rec_coord, input_coord)
                #self.train_points_clamped_mse(torch.clamp(rec_coord, self.depth_range[0], self.depth_range[1]), input_coord)
                #self.train_points_clamped_mae(torch.clamp(rec_coord, self.depth_range[0], self.depth_range[1]), input_coord)
                log[f"{split}/img_mse"] = self.train_img_mse
                log[f"{split}/img_mae"] = self.train_img_mae
                log[f"{split}/points_mse"] = self.train_points_mse
                log[f"{split}/points_mae"] = self.train_points_mae
                #log[f"{split}/points_clamped_mse"] = self.train_points_clamped_mse
                #log[f"{split}/points_clamped_mae"] = self.train_points_clamped_mae
            
            if split=="val":
                self.val_img_mse(reconstructions[:, 0:1].contiguous(), inputs.contiguous())
                self.val_img_mae(reconstructions[:, 0:1].contiguous(), inputs.contiguous()) 
                self.val_points_mse(rec_coord, input_coord)
                self.val_points_mae(rec_coord, input_coord)
                #self.val_points_clamped_mse(torch.clamp(rec_coord, self.depth_range[0], self.depth_range[1]), input_coord)
                #self.val_points_clamped_mae(torch.clamp(rec_coord, self.depth_range[0], self.depth_range[1]), input_coord)
                log[f"{split}/img_mse"] = self.val_img_mse
                log[f"{split}/img_mae"] = self.val_img_mae
                log[f"{split}/points_mse"] = self.val_points_mse
                log[f"{split}/points_mae"] = self.val_points_mae
                #log[f"{split}/points_clamped_mse"] = self.val_points_clamped_mse
                #log[f"{split}/points_clamped_mae"] = self.val_points_clamped_mae

            if predicted_indices is not None:
                assert self.n_classes is not None
                with torch.no_grad():
                    perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_classes)
                log[f"{split}/perplexity"] = perplexity
                log[f"{split}/cluster_usage"] = cluster_usage
            return loss, log

        # update discriminator (input: img, mask, coord, [cond])
        if optimizer_idx == 1:
            disc_inputs, disc_recons = inputs.contiguous().detach(), reconstructions.contiguous().detach()
            if self.mask_factor > 0 and masks is not None:
                disc_inputs = torch.cat((disc_inputs, masks.contiguous().detach()), dim=1)
            if self.geo_factor > 0:
                disc_inputs = torch.cat((disc_inputs, input_coord[:, :2].contiguous()), dim=1)  # BEV only
                disc_recons = torch.cat((disc_recons, rec_coord[:, :2].contiguous()), dim=1)    # BEV only
            if cond is not None:
                disc_inputs = torch.cat((disc_inputs, cond), dim=1)
                disc_recons = torch.cat((disc_recons, cond), dim=1)
            logits_real = self.discriminator(disc_inputs)
            logits_fake = self.discriminator(disc_recons)

            # gan loss
            d_loss = self.disc_loss(logits_real, logits_fake) * disc_factor

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()}

            return d_loss, log

"""