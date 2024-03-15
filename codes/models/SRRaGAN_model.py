import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss
logger = logging.getLogger('base')

import pywt 
import models.SWT as SWT
import numpy as np
import piq

class SRRaGANModel(BaseModel):
    def __init__(self, opt):
        super(SRRaGANModel, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)  # D
            self.netG.train()
            self.netD.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
                self.l_pix_w_lh = train_opt['pixel_weight_lh']
                self.l_pix_w_hl = train_opt['pixel_weight_hl']
                self.l_pix_w_hh = train_opt['pixel_weight_hh']
                self.l_pix_w_lh2 = train_opt['pixel_weight_lh2']
                self.l_pix_w_hl2 = train_opt['pixel_weight_hl2']
                self.l_pix_w_hh2 = train_opt['pixel_weight_hh2']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            ###setting wavelet filter name and wavelet decomposition level
            self.filter_name = train_opt['wavelet_filter']
            self.level = train_opt['wavelet_level']

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                #self.netF = networks.define_F(opt, use_bn=False).to(self.device)  
                #### instead of using VGG perceptual loss, we utilized DISTS from piq, further info pls refer to "https://piq.readthedocs.io/en/latest/"
                self.netF = piq.DISTS()

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters are for WGAN
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            if train_opt['gan_type'] == 'wgan-gp':
                self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                # gradient penalty loss
                self.cri_gp = GradientPenaltyLoss(device=self.device).to(self.device)
                self.l_gp_w = train_opt['gp_weigth']

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        self.print_network()

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data['LR'].to(self.device)

        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)

            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()

        self.fake_H = self.netG(self.var_L)

        # wavelet init
        wavelet = pywt.Wavelet(self.filter_name)
            
        dlo = wavelet.dec_lo
        an_lo = np.divide(dlo, sum(dlo))
        an_hi = wavelet.dec_hi
        rlo = wavelet.rec_lo
        syn_lo = 2*np.divide(rlo, sum(rlo))
        syn_hi = wavelet.rec_hi

        filters = pywt.Wavelet('wavelet_normalized', [an_lo, an_hi, syn_lo, syn_hi])
        self.sfm = SWT.SWTForward(1, filters, 'periodic').to(self.device)
        self.ifm = SWT.SWTInverse(filters, 'periodic').to(self.device)

        ## wavelet bands of sr image
        sr_img_y       = 16.0 + (self.fake_H[:,0:1,:,:]*65.481 + self.fake_H[:,1:2,:,:]*128.553 + self.fake_H[:,2:,:,:]*24.966)
        wavelet_sr     = self.sfm(sr_img_y)[0]

        self.LL_band   = wavelet_sr[:,0:1, :, :]
        self.LH_band   = wavelet_sr[:,1:2, :, :]
        self.HL_band   = wavelet_sr[:,2:3, :, :]
        self.HH_band   = wavelet_sr[:,3:, :, :]

        self.combined_HF_bands     = torch.cat((self.LH_band, self.HL_band, self.HH_band), axis = 1)       

        if self.level == 2:

            wavelet_sr2   = self.sfm(self.LL_band)[0]
            self.LL_band2 = wavelet_sr2[:,0:1, :,:]
            self.LH_band2  = wavelet_sr2[:,1:2, :, :]
            self.HL_band2  = wavelet_sr2[:,2:3, :, :]
            self.HH_band2  = wavelet_sr2[:,3:, :, :]

            self.combined_LH = torch.cat((self.LH_band, self.LH_band2), axis=1)
            self.combined_HL = torch.cat((self.HL_band, self.HL_band2), axis=1)
            self.combined_HH = torch.cat((self.HH_band, self.HH_band2), axis=1)

            self.combined_HF_bands     = torch.cat((self.combined_LH, self.combined_HL, self.combined_HH), axis = 1)       

        ## wavelet bands of hr image
        hr_img_y       = 16.0 + (self.var_H[:,0:1,:,:]*65.481 + self.var_H[:,1:2,:,:]*128.553 + self.var_H[:,2:,:,:]*24.966)
        wavelet_hr     = self.sfm(hr_img_y)[0]

        self.LL_band_hr   = wavelet_hr[:,0:1, :, :]
        self.LH_band_hr   = wavelet_hr[:,1:2, :, :]
        self.HL_band_hr   = wavelet_hr[:,2:3, :, :]
        self.HH_band_hr   = wavelet_hr[:,3:, :, :]

        self.combined_HF_bands_hr     = torch.cat((self.LH_band_hr, self.HL_band_hr, self.HH_band_hr), axis = 1)       

        if self.level == 2:

            wavelet_hr2  = self.sfm(self.LL_band_hr)[0]

            self.LL_band_hr2   = wavelet_hr2[:,0:1, :, :]
            self.LH_band_hr2   = wavelet_hr2[:,1:2, :, :]
            self.HL_band_hr2  = wavelet_hr2[:,2:3, :, :]
            self.HH_band_hr2   = wavelet_hr2[:,3:, :, :]

            self.combined_LH_hr = torch.cat((self.LH_band_hr, self.LH_band_hr2), axis=1)
            self.combined_HL_hr = torch.cat((self.HL_band_hr, self.HL_band_hr2), axis=1)
            self.combined_HH_hr = torch.cat((self.HH_band_hr, self.HH_band_hr2), axis=1)

            self.combined_HF_bands_hr     = torch.cat((self.combined_LH_hr, self.combined_HL_hr, self.combined_HH_hr), axis = 1)       

        if self.level == 1:
            l_g_total = 0
            if step % self.D_update_ratio == 0 and step > self.D_init_iters:
                if self.cri_pix:  # pixel loss
                    l_g_pix = self.l_pix_w * self.cri_pix(self.LL_band, self.LL_band_hr)
                    l_g_pix_lh = self.l_pix_w_lh * self.cri_pix(self.LH_band, self.LH_band_hr)
                    l_g_pix_hl = self.l_pix_w_hl * self.cri_pix(self.HL_band, self.HL_band_hr)
                    l_g_pix_hh = self.l_pix_w_hh * self.cri_pix(self.HH_band, self.HH_band_hr)
                    l_g_total = l_g_total + l_g_pix + l_g_pix_lh + l_g_pix_hl + l_g_pix_hh
                    print("LL: ", l_g_pix, "LH: ", l_g_pix_lh, "HL: ", l_g_pix_hl, "HH: ", l_g_pix_hh)

                if self.cri_fea:  # feature loss
                    #real_fea = self.netF(self.var_H).detach()
                    #fake_fea = self.netF(self.fake_H)
                    #l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                    l_g_fea = self.l_fea_w * self.netF(self.var_H.detach(), self.fake_H)
                    l_g_total += l_g_fea
                # G gan + cls loss
                pred_g_fake = self.netD(self.combined_HF_bands)
                pred_d_real = self.netD(self.combined_HF_bands_hr).detach()

                l_g_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                        self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                l_g_total += l_g_gan

                l_g_total.backward()
                self.optimizer_G.step()


        elif self.level == 2:
            l_g_total = 0
            if step % self.D_update_ratio == 0 and step > self.D_init_iters:
                if self.cri_pix:  # pixel loss
                    l_g_pix = self.l_pix_w * (self.cri_pix(self.LL_band, self.LL_band_hr))
                    l_g_pix_lh = self.l_pix_w_lh * (self.cri_pix(self.LH_band, self.LH_band_hr)) + self.l_pix_w_lh2 * (self.cri_pix(self.LH_band2, self.LH_band_hr2))
                    l_g_pix_hl = self.l_pix_w_hl * (self.cri_pix(self.HL_band, self.HL_band_hr)) + self.l_pix_w_hl2 * (self.cri_pix(self.HL_band2, self.HL_band_hr2))
                    l_g_pix_hh = self.l_pix_w_hh * (self.cri_pix(self.HH_band, self.HH_band_hr)) + self.l_pix_w_hh2 * self.cri_pix(self.HH_band2, self.HH_band_hr2)
                    l_g_total = l_g_total + l_g_pix + l_g_pix_lh + l_g_pix_hl + l_g_pix_hh
                    print("LL: ", l_g_pix, "LH: ", l_g_pix_lh, "HL: ", l_g_pix_hl, "HH: ", l_g_pix_hh)
                    
                if self.cri_fea:  # feature loss for only high-frequency subbands
                    l_g_fea = self.l_fea_w * self.netF(self.var_H.detach(), self.fake_H)
                    l_g_total += l_g_fea
                # G gan + cls loss
                pred_g_fake = self.netD(self.combined_HF_bands)
                pred_d_real = self.netD(self.combined_HF_bands_hr).detach()

                l_g_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                        self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                l_g_total += l_g_gan

                l_g_total.backward()
                self.optimizer_G.step()

        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total = 0

        pred_d_real = self.netD(self.combined_HF_bands_hr)
        pred_d_fake = self.netD(self.combined_HF_bands.detach())  # detach to avoid BP to G

        l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
        l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)

        l_d_total = (l_d_real + l_d_fake) / 2

        if self.opt['train']['gan_type'] == 'wgan-gp':
            batch_size = self.var_ref.size(0)
            if self.random_pt.size(0) != batch_size:
                self.random_pt.resize_(batch_size, 1, 1, 1)
            self.random_pt.uniform_()  # Draw random interpolation points
            interp = self.random_pt * self.fake_H.detach() + (1 - self.random_pt) * self.var_ref
            interp.requires_grad = True
            interp_crit = self.netD(interp)
            l_d_gp = self.l_gp_w * self.cri_gp(interp, interp_crit)
            l_d_total += l_d_gp

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # G
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()
        # D
        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()

        if self.opt['train']['gan_type'] == 'wgan-gp':
            self.log_dict['l_d_gp'] = l_d_gp.item()
        # D outputs
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)

            logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

            if not self.cri_fea:  # F, Perceptual Network  not ekledim 20221214
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                    self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)

                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
