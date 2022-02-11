import sys


from agents.loss import LossRecorder
from utils.camera import K_partition, partition_image_and_matrix, CameraPOs_as_torch_partitioned, resize_matrix, interpolate_cameras
from dataset.dataset import MVSDataset
from configs.parameter import Params
#from graphs.losses.biscale_loss import HyperLoss_2d, PSNRLoss, SSIMLoss, LPIPSLoss
from graphs.render.render_base import RenderMachine
from graphs.render.rasterizer import generate_tri_mesh

from utils.scene import save2ply
from utils.visualize import create_mask_image, create_alpha_channel
from tensorboardX import SummaryWriter
import pdb
import time
import shutil
import logging
import torch
import torch.nn as nn

from PIL import Image, ImageDraw

from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import os

import numpy as np
import cv2
import scipy.misc
import math
import matplotlib.pyplot as plt

import time




class VisualizeImages():
    def __init__(self, params):
        super(VisualizeImages, self).__init__()

        self.params = params

    def create_file(self, model_i):

        if(self.params.mode is 'render_novel_view'):
            self.file_root_interpolate_predict = os.path.join(self.params.root_file, self.params.renderResultRoot,
                                                         str(self.params.modelList[model_i]), 'interpolate_predict')
            if not os.path.exists(self.file_root_interpolate_predict):
                os.makedirs(self.file_root_interpolate_predict)

            self.file_root_interpolate_zbuf = os.path.join(self.params.root_file, self.params.renderResultRoot,
                                                      str(self.params.modelList[model_i]), 'interpolate_zbuf')
            if not os.path.exists(self.file_root_interpolate_zbuf):
                os.makedirs(self.file_root_interpolate_zbuf)

            self.file_root_interpolate_mask = os.path.join(self.params.root_file, self.params.renderResultRoot,
                                                      str(self.params.modelList[model_i]), 'interpolate_mask')
            if not os.path.exists(self.file_root_interpolate_mask):
                os.makedirs(self.file_root_interpolate_mask)

        elif (self.params.mode is 'debug_3') or (self.params.mode is 'reconstruct'):
            self.file_root_gt = os.path.join(self.params.root_file, self.params.renderResultRoot,
                                        str(self.params.modelList[model_i]), 'gt')
            if not os.path.exists(self.file_root_gt):
                os.makedirs(self.file_root_gt)

            self.file_root_mask = os.path.join(self.params.root_file, self.params.renderResultRoot,
                                          str(self.params.modelList[model_i]), 'mask')
            if not os.path.exists(self.file_root_mask):
                os.makedirs(self.file_root_mask)

            self.file_root_zbuf = os.path.join(self.params.root_file, self.params.renderResultRoot,
                                          str(self.params.modelList[model_i]), 'zbuf')
            if not os.path.exists(self.file_root_zbuf):
                os.makedirs(self.file_root_zbuf)

            self.file_root_predict = os.path.join(self.params.root_file, self.params.renderResultRoot,
                                             str(self.params.modelList[model_i]), 'predict')
            if not os.path.exists(self.file_root_predict):
                os.makedirs(self.file_root_predict)

            #######################################################################
            # self.file_root_mask_numpy = os.path.join(self.params.root_file, self.params.renderResultRoot_numpy,
            #                                     str(self.params.modelList[model_i]), 'mask')
            # if not os.path.exists(self.file_root_mask_numpy):
            #     os.makedirs(self.file_root_mask_numpy)
            #
            # self.file_root_zbuf_numpy = os.path.join(self.params.root_file, self.params.renderResultRoot_numpy,
            #                                     str(self.params.modelList[model_i]), 'zbuf')
            # if not os.path.exists(self.file_root_zbuf_numpy):
            #     os.makedirs(self.file_root_zbuf_numpy)


    def initialize_empty_images(self, num_view):
        if (self.params.mode is 'render_novel_view'):
            if self.params.save_mode == 'alpha':
                self.IMG0 = np.empty((num_view, int(self.params.img_h / self.params.compress_ratio_total),
                                 int(self.params.img_w / self.params.compress_ratio_total), 4), dtype=np.float64)
                self.IMG1 = np.empty((num_view, int(self.params.img_h / self.params.compress_ratio_total),
                                 int(self.params.img_w / self.params.compress_ratio_total), 4), dtype=np.float64)
            else:
                self.IMG0 = np.empty((num_view, int(self.params.img_h / self.params.compress_ratio_total),
                                 int(self.params.img_w / self.params.compress_ratio_total), 3), dtype=np.float64)
                self.IMG1 = np.empty((num_view, int(self.params.img_h / self.params.compress_ratio_total),
                                 int(self.params.img_w / self.params.compress_ratio_total), 3), dtype=np.float64)
            self.MASK = np.empty((num_view, int(self.params.img_h / self.params.compress_ratio_total),
                             int(self.params.img_w / self.params.compress_ratio_total)), dtype=np.float64)
            self.ZBUF = np.empty((num_view, int(self.params.img_h / self.params.compress_ratio_total),
                             int(self.params.img_w / self.params.compress_ratio_total), 3), dtype=np.float64)

        elif (self.params.mode is 'debug_3') or (self.params.mode is 'reconstruct'):
            if self.params.save_mode == 'alpha':
                self.IMG_gt = np.empty((num_view, int(self.params.img_h / self.params.compress_ratio_total),
                                   int(self.params.img_w / self.params.compress_ratio_total), 4), dtype=np.float64)
                self.IMG0 = np.empty((num_view, int(self.params.img_h / self.params.compress_ratio_total),
                                 int(self.params.img_w / self.params.compress_ratio_total), 4), dtype=np.float64)
                self.IMG1 = np.empty((num_view, int(self.params.img_h / self.params.compress_ratio_total),
                                 int(self.params.img_w / self.params.compress_ratio_total), 4), dtype=np.float64)
            else:
                self.IMG_gt = np.empty((num_view, int(self.params.img_h / self.params.compress_ratio_total),
                                   int(self.params.img_w / self.params.compress_ratio_total), 3), dtype=np.float64)
                self.IMG0 = np.empty((num_view, int(self.params.img_h / self.params.compress_ratio_total),
                                 int(self.params.img_w / self.params.compress_ratio_total), 3), dtype=np.float64)
                self.IMG1 = np.empty((num_view, int(self.params.img_h / self.params.compress_ratio_total),
                                 int(self.params.img_w / self.params.compress_ratio_total), 3), dtype=np.float64)
            self.MASK = np.empty((num_view, int(self.params.img_h / self.params.compress_ratio_total),
                             int(self.params.img_w / self.params.compress_ratio_total)), dtype=np.float64)
            self.ZBUF = np.empty((num_view, int(self.params.img_h / self.params.compress_ratio_total),
                             int(self.params.img_w / self.params.compress_ratio_total), 3), dtype=np.float64)

    def combine_images(self, partition_j, mask, images, zbuf, id_end, id_init, parti_imgs):
        if (self.params.mode is 'render_novel_view'):
            start_h_idx = math.floor(partition_j / self.params.compress_ratio_w)
            start_w_idx = (partition_j % self.params.compress_ratio_w)

            start_h_pix = start_h_idx * int(
                self.params.img_h / (self.params.compress_ratio_h * self.params.compress_ratio_total))
            start_w_pix = start_w_idx * int(
                self.params.img_w / (self.params.compress_ratio_w * self.params.compress_ratio_total))

            final_h_pix = start_h_pix + int(
                self.params.img_h / (self.params.compress_ratio_h * self.params.compress_ratio_total))
            final_w_pix = start_w_pix + int(
                self.params.img_w / (self.params.compress_ratio_w * self.params.compress_ratio_total))

            mask_image = create_mask_image(mask)
            alpha_channel = create_alpha_channel(mask)

            for i in range(id_end - id_init):
                # change the saving format.
                if self.params.save_mode == 'gray':
                    self.IMG0[i, start_h_pix: final_h_pix, start_w_pix: final_w_pix, :] = 0.5 + images[i, :3, :,
                                                                                           :].detach().permute(1, 2,
                                                                                                               0).cpu().numpy()
                    scipy.misc.imsave(os.path.join(self.file_root_interpolate_predict,
                                                   'init_%s_part%s.jpg' % (str(i + id_init).zfill(8), str(partition_j))),
                                      0.5 + images[i, :3, :, :].detach().permute(1, 2, 0).cpu().numpy())
                elif self.params.save_mode == 'white':
                    self.IMG0[i, start_h_pix: final_h_pix, start_w_pix: final_w_pix, :] = mask_image[i] + images[i, :3, :,
                                                                                                     :].detach().permute(1,
                                                                                                                         2,
                                                                                                                         0).cpu().numpy()
                    scipy.misc.imsave(os.path.join(self.file_root_interpolate_predict,
                                                   'init_%s_part%s.jpg' % (str(i + id_init).zfill(8), str(partition_j))),
                                      mask_image[i] + images[i, :3, :, :].detach().permute(1, 2, 0).cpu().numpy())
                elif self.params.save_mode == 'alpha':
                    self.IMG0[i, start_h_pix: final_h_pix, start_w_pix: final_w_pix, :] = np.concatenate(
                        (0.5 + images[i, :3, :, :].detach().permute(1, 2, 0).cpu().numpy(), alpha_channel[i]), -1)
                    scipy.misc.imsave(os.path.join(self.file_root_interpolate_predict,
                                                   'init_%s_part%s.png' % (str(i + id_init).zfill(8), str(partition_j))),
                                      np.concatenate((0.5 + images[i, :3, :, :].detach().permute(1, 2, 0).cpu().numpy(),
                                                      alpha_channel[i]), -1))

            for i in range(id_end - id_init):

                # change the saving format.
                self.MASK[i, start_h_pix: final_h_pix, start_w_pix: final_w_pix] = 0.0 + mask[i, 0].cpu().numpy()
                scipy.misc.imsave(os.path.join(self.file_root_interpolate_mask,
                                               '%s_part%s.jpg' % (str(i + id_init).zfill(8), str(partition_j))),
                                  0.0 + mask[i, 0].cpu().numpy())

            for i in range(id_end - id_init):
                zbuf_out = 255 * (-self.params.zbuf_min + torch.clamp(zbuf[i], self.params.zbuf_min,
                                                                      self.params.zbuf_max).detach().permute(1, 2,
                                                                                                             0).cpu().numpy()) / (
                                       self.params.zbuf_max - self.params.zbuf_min)

                zbuf_out = np.array(zbuf_out, np.uint8)

                im_color = cv2.applyColorMap(zbuf_out, cv2.COLORMAP_JET)

                # change the saving format.
                self.ZBUF[i, start_h_pix: final_h_pix, start_w_pix: final_w_pix, :] = im_color
                scipy.misc.imsave(os.path.join(self.file_root_interpolate_zbuf, '%s_part%s.jpg' % (
                    str(i + id_init).zfill(8), str(partition_j))), im_color)


        elif (self.params.mode is 'debug_3') or (self.params.mode is 'reconstruct'):

            start_h_idx = math.floor(partition_j / self.params.compress_ratio_w)
            start_w_idx = (partition_j % self.params.compress_ratio_w)

            start_h_pix = start_h_idx * int(
                self.params.img_h / (self.params.compress_ratio_h * self.params.compress_ratio_total))
            start_w_pix = start_w_idx * int(
                self.params.img_w / (self.params.compress_ratio_w * self.params.compress_ratio_total))

            final_h_pix = start_h_pix + int(
                self.params.img_h / (self.params.compress_ratio_h * self.params.compress_ratio_total))
            final_w_pix = start_w_pix + int(
                self.params.img_w / (self.params.compress_ratio_w * self.params.compress_ratio_total))

            # if self.params.compute_psnr_loss:
            #     loss = self.psnr_loss(mask * images, mask * torch.from_numpy(parti_imgs[:, partition_j, :, :, :]).type(mask.dtype).to(self.device))
            # if self.params.compute_psnr_loss:
            #     print('loss:', loss)

            mask_image = create_mask_image(mask)
            alpha_channel = create_alpha_channel(mask)

            for i in range(id_end - id_init):
                if self.params.save_mode == 'gray':
                    self.IMG0[i, start_h_pix: final_h_pix, start_w_pix: final_w_pix, :] = 0.5 + images[i, :3, :,
                                                                                                :].detach().permute(1,
                                                                                                                    2,
                                                                                                                    0).cpu().numpy()
                    scipy.misc.imsave(os.path.join(self.file_root_predict,
                                                   'final_%s_part%s.jpg' % (
                                                   str(self.params.testViewList[i + id_init]).zfill(8), str(partition_j))),
                                      0.5 + images[i, :3, :, :].detach().permute(1, 2, 0).cpu().numpy())
                elif self.params.save_mode == 'white':
                    self.IMG0[i, start_h_pix: final_h_pix, start_w_pix: final_w_pix, :] = mask_image[i] + images[i, :3,
                                                                                                          :,
                                                                                                          :].detach().permute(
                                                                                                                            1,
                                                                                                                            2,
                                                                                                                            0).cpu().numpy()
                    scipy.misc.imsave(os.path.join(self.file_root_predict,
                                                   'final_%s_part%s.jpg' % (
                                                   str(self.params.testViewList[i + id_init]).zfill(8), str(partition_j))),
                                      mask_image[i] + images[i, :3, :, :].detach().permute(1, 2, 0).cpu().numpy())
                elif self.params.save_mode == 'alpha':
                    self.IMG0[i, start_h_pix: final_h_pix, start_w_pix: final_w_pix, :] = np.concatenate(
                        (0.5 + images[i, :3, :, :].detach().permute(1, 2, 0).cpu().numpy(), alpha_channel[i]), -1)
                    scipy.misc.imsave(os.path.join(self.file_root_predict,
                                                   'final_%s_part%s.png' % (
                                                   str(self.params.testViewList[i + id_init]).zfill(8), str(partition_j))),
                                      np.concatenate((0.5 + images[i, :3, :, :].detach().permute(1, 2, 0).cpu().numpy(),
                                                      alpha_channel[i]), -1))

            image_gt = parti_imgs[id_init:id_end, partition_j, :, :, :]
            image_gt = image_gt * mask.cpu().numpy()
            image_gt = torch.from_numpy(image_gt).permute(0, 2, 3, 1).numpy()
            for i in range(id_end - id_init):
                if self.params.save_mode == 'gray':
                    self.IMG_gt[i, start_h_pix: final_h_pix, start_w_pix: final_w_pix, :] = 0.5 + image_gt[i, :, :, :]
                    scipy.misc.imsave(os.path.join(self.file_root_gt,
                                                   '%s_part%s.jpg' % (
                                                   str(self.params.testViewList[i + id_init]).zfill(8), str(partition_j))),
                                      0.5 + image_gt[i, :, :, :])
                elif self.params.save_mode == 'white':
                    self.IMG_gt[i, start_h_pix: final_h_pix, start_w_pix: final_w_pix, :] = mask_image[i] + image_gt[i, :, :, :]
                    scipy.misc.imsave(os.path.join(self.file_root_gt,
                                                   '%s_part%s.jpg' % (
                                                   str(self.params.testViewList[i + id_init]).zfill(8), str(partition_j))),
                                      mask_image[i] + image_gt[i, :, :, :])
                elif self.params.save_mode == 'alpha':
                    self.IMG_gt[i, start_h_pix: final_h_pix, start_w_pix: final_w_pix, :] = np.concatenate(
                        (0.5 + image_gt[i, :, :, :], alpha_channel[i]), -1)
                    scipy.misc.imsave(os.path.join(self.file_root_gt,
                                                   '%s_part%s.png' % (
                                                   str(self.params.testViewList[i + id_init]).zfill(8), str(partition_j))),
                                      np.concatenate((0.5 + image_gt[i, :, :, :], alpha_channel[i]), -1))

            if (self.params.save_depth):
                for i in range(id_end - id_init):
                    # mask_finer = self.MVSDataset_test.mask_list_models[model_i, i + id_init][None, ...].to(
                    #     self.device)
                    np.save(os.path.join(self.file_root_mask_numpy, '%s_part%s' % (
                        str(self.params.testViewList[i + id_init]).zfill(8), str(partition_j))),
                            mask[i, 0].cpu().numpy())  # save shape: (H,W)

            for i in range(id_end - id_init):
                self.MASK[i, start_h_pix: final_h_pix, start_w_pix: final_w_pix] = 0.0 + mask[i, 0].cpu().numpy()
                scipy.misc.imsave(os.path.join(self.file_root_mask, '%s_part%s.jpg' % (
                    str(self.params.testViewList[i + id_init]).zfill(8), str(partition_j))),
                                  0.0 + mask[i, 0].cpu().numpy())

            if (self.params.save_depth):
                for i in range(id_end - id_init):
                    np.save(os.path.join(self.file_root_zbuf_numpy, '%s_part%s' % (
                        str(self.params.testViewList[i + id_init]).zfill(8), str(partition_j))),
                            zbuf[i].detach().cpu().numpy())  # save shape: (1,H,W)

            for i in range(id_end - id_init):
                # zbuf_finer = self.MVSDataset_test.zbuf_list_models[model_i, i + id_init][None, ...].to(
                #     self.device)
                # clamp_length = self.params.D_length / 8
                zbuf_out = 255 * (-self.params.zbuf_min + torch.clamp(zbuf[i], self.params.zbuf_min,
                                                                      self.params.zbuf_max).detach().permute(1,
                                                                                                             2,
                                                                                                             0).cpu().numpy()) / (
                                   self.params.zbuf_max - self.params.zbuf_min)
                # zbuf_out = 255 * (clamp_length + torch.clamp(zbuf_finer - zbuf[i], -clamp_length, clamp_length).detach().permute(1,2,0).cpu().numpy()) / (2 * clamp_length)
                # zbuf_out = 255 * (-self.params.zbuf_min + torch.clamp(zbuf_finer, self.params.zbuf_min, self.params.zbuf_max).detach().permute(1, 2,0).cpu().numpy()) / (self.params.zbuf_max - self.params.zbuf_min)
                # zbuf_out = 255 * (self.params.D_length + torch.clamp(zbuf[i], -self.params.D_length, self.params.D_length).detach().permute(1,2,0).cpu().numpy()) / (2 * self.params.D_length)
                # zbuf_out = 255 * (clamp_length + torch.clamp(zbuf[i], -clamp_length, clamp_length).detach().permute(1,2,0).cpu().numpy()) / (2 * clamp_length)

                zbuf_out = np.array(zbuf_out, np.uint8)

                im_color = cv2.applyColorMap(zbuf_out, cv2.COLORMAP_JET)
                self.ZBUF[i, start_h_pix: final_h_pix, start_w_pix: final_w_pix, :] = im_color
                scipy.misc.imsave(os.path.join(self.file_root_zbuf, '%s_part%s.jpg' % (
                    str(self.params.testViewList[i + id_init]).zfill(8), str(partition_j))), im_color)

    def save_total_image(self, id_end, id_init):

        if (self.params.mode is 'render_novel_view'):
            for i in range(id_end - id_init):
                # change the saving format.
                if self.params.save_mode == 'alpha':
                    scipy.misc.imsave(
                        os.path.join(self.file_root_interpolate_predict, 'afull_init_%s.png' % str(i + id_init).zfill(8)),
                        self.IMG0[i])
                    if (self.params.use_2d_network):
                        scipy.misc.imsave(
                            os.path.join(self.file_root_interpolate_predict, 'afull_final_%s.png' % str(i + id_init).zfill(8)),
                            self.IMG1[i])
                else:
                    scipy.misc.imsave(
                        os.path.join(self.file_root_interpolate_predict, 'afull_init_%s.jpg' % str(i + id_init).zfill(8)),
                        self.IMG0[i])
                    if (self.params.use_2d_network):
                        scipy.misc.imsave(
                            os.path.join(self.file_root_interpolate_predict, 'afull_final_%s.jpg' % str(i + id_init).zfill(8)),
                            self.IMG1[i])
                scipy.misc.imsave(os.path.join(self.file_root_interpolate_mask, 'full_%s.jpg' % str(i + id_init).zfill(8)),
                                  self.MASK[i])
                scipy.misc.imsave(os.path.join(self.file_root_interpolate_zbuf, 'full_%s.jpg' % str(i + id_init).zfill(8)),
                                  self.ZBUF[i])




        elif (self.params.mode is 'debug_3') or (self.params.mode is 'reconstruct'):
            for i in range(id_end - id_init):
                if self.params.save_mode == 'alpha':
                    scipy.misc.imsave(os.path.join(self.file_root_predict,
                                                   'afull_init_%s.png' % str(self.params.testViewList[i + id_init]).zfill(
                                                       8)), self.IMG0[i])
                    scipy.misc.imsave(
                        os.path.join(self.file_root_gt, 'full_%s.png' % str(self.params.testViewList[i + id_init]).zfill(8)),
                        self.IMG_gt[i])
                    if (self.params.use_2d_network):
                        scipy.misc.imsave(os.path.join(self.file_root_predict, 'afull_final_%s.png' % str(
                            self.params.testViewList[i + id_init]).zfill(8)), self.IMG1[i])
                else:
                    scipy.misc.imsave(os.path.join(self.file_root_predict,
                                                   'afull_init_%s.jpg' % str(self.params.testViewList[i + id_init]).zfill(
                                                       8)), self.IMG0[i])
                    scipy.misc.imsave(
                        os.path.join(self.file_root_gt, 'full_%s.jpg' % str(self.params.testViewList[i + id_init]).zfill(8)),
                        self.IMG_gt[i])
                    if (self.params.use_2d_network):
                        scipy.misc.imsave(os.path.join(self.file_root_predict, 'afull_final_%s.jpg' % str(
                            self.params.testViewList[i + id_init]).zfill(8)), self.IMG1[i])
                scipy.misc.imsave(
                    os.path.join(self.file_root_mask, 'full_%s.jpg' % str(self.params.testViewList[i + id_init]).zfill(8)),
                    self.MASK[i])
                scipy.misc.imsave(
                    os.path.join(self.file_root_zbuf, 'full_%s.jpg' % str(self.params.testViewList[i + id_init]).zfill(8)),
                    self.ZBUF[i])

    def generate_gif(self, num_view_interpolate):
        images = []
        for i in range(0, num_view_interpolate, 1):
            # change the saving format.

            if self.params.save_mode == 'alpha':
                im = Image.open(os.path.join(self.file_root_interpolate_predict,
                                             'afull_init_%s.png' % str(i).zfill(8)))
            else:
                im = Image.open(os.path.join(self.file_root_interpolate_predict,
                                             'afull_init_%s.jpg' % str(i).zfill(8)))
            images.append(im)

        images[0].save(os.path.join(self.file_root_interpolate_predict, 'render_slow.gif'),
                       save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)
        images[0].save(os.path.join(self.file_root_interpolate_predict, 'render_fast.gif'),
                       save_all=True, append_images=images[1:], optimize=False, duration=50, loop=0)