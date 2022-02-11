"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import sys

sys.path.append("../")

# sys.path.append("/home/jinzhi/hdd10T/aMyLab/projects/Giant/v_0/jinzhi/v_0")
from agents.loss import LossRecorder
from utils.camera import K_partition, partition_image_and_matrix, CameraPOs_as_torch_partitioned, resize_matrix
from dataset.dataset import MVSDataset
from configs.parameter import Params
from graphs.render.render_base import RenderMachine
from graphs.render.rasterizer import generate_tri_mesh, generate_pointcloud
from agents.visualize import VisualizeImages

from utils.scene import save2ply
from utils.visualize import create_mask_image, create_alpha_channel
from agents.reconstruct import ReconstructGraph
from graphs.models.network import ReptileEmbedding

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

class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, params):
        self.params = params
        self.logger = logging.getLogger("Agent")
        self.debug_flag = True

        self.t_begin = time.time()
        self.manual_seed = random.randint(1, 10000)
        print("seed: ", self.manual_seed)
        random.seed(self.manual_seed)

        use_cuda = self.params.use_cuda

        if (use_cuda and torch.cuda.is_available()):
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.manual_seed)
            print("Program will run on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            print("Program will run on *****CPU***** ")

        torch.multiprocessing.set_sharing_strategy('file_system')

        self.current_epoch = 0
        self.current_iteration = 0
        self.summary_writer = SummaryWriter(log_dir=self.params.summary_dir, comment='biscale')


        self.MVSDataset = MVSDataset(self.params)
        self.MVSDataset_test = MVSDataset(self.params, type_data = 'test')

        self.camera_intrisic = (self.MVSDataset.cameraKO4s_models[0,0,0,0].item() * self.MVSDataset.cameraKO4s_models[0,0,1,1].item() * self.params.compress_ratio_h * self.params.compress_ratio_w) ** 0.5
        self.camera_intrisic_test = (self.MVSDataset_test.cameraKO4s_models[0,0,0,0].item() * self.MVSDataset_test.cameraKO4s_models[0,0,1,1].item() * self.params.compress_ratio_h * self.params.compress_ratio_w) ** 0.5

        if (self.params.backward_gradient_mode is 'meta'):
            self.render_machine_meta = RenderMachine(params=self.params, device=self.device)
            self.meta_optimizer_state = None
            self.render_machine = self.render_machine_meta.clone()
        else:
            self.render_machine = RenderMachine(params=self.params, device=self.device)

        self.data_initialize()


        self.mse = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        # if self.params.compute_psnr_loss:
        #     self.psnr_loss = PSNRLoss(params = self.params)
        # self.ssim_loss = SSIMLoss(params = self.params)
        # self.lpips_loss = LPIPSLoss(params = self.params)

        self.lr = self.params.lr_net_2d
        self.update_settings()

        self.load_checkpoint(filename = self.params.load_checkpoint_dir)
        self.loss_recorder = LossRecorder(self.params)
        self.image_visualizer = VisualizeImages(self.params)
        self.graph_reconstruct = ReconstructGraph(self.params)

    def data_initialize(self):

        if (self.params.rasterizer_mode is 'pointcloud'):
            self.pointclouds_list = []
        elif (self.params.rasterizer_mode is 'mesh'):
            self.meshes_list = []
        self.basic_len = []

        if(self.params.backward_gradient_mode is 'meta'):
            # self.z_embeddings_list_meta = ReptileEmbedding()
            # if (self.params.use_lighting_embedding):
            #     self.lighting_embedding_list_meta = ReptileEmbedding()
            self.z_embeddings_list = nn.ModuleList()
            if (self.params.use_lighting_embedding):
                self.lighting_embedding_list = nn.ModuleList()
        else:
            self.z_embeddings_list = nn.ModuleList()
            if(self.params.use_lighting_embedding):
                self.lighting_embedding_list = nn.ModuleList()

        gt_models = self.MVSDataset.gt_models
        node_num = self.params.node_num

        count = 1
        sum_num = 0
        for i in range(self.params.init_stage):
            sum_num += self.params.node_num[i]
            #count = count * self.params.stage_base_num

        for i in range(len(self.params.modelList)):

            if(self.params.load_preprocess_index):
                file_root_reconstruction = os.path.join(self.params.root_file, self.params.renderResultRoot_preprocess,
                                                        str(self.params.modelList[i]), 'points_filter')
                points_mask = np.load(os.path.join(file_root_reconstruction, 'mask.npy'))
                if (self.params.rasterizer_mode is 'pointcloud'):
                    point_clouds = generate_pointcloud(gt_models[i][0][points_mask], gt_models[i][1][points_mask],
                                                       gt_models[i][2][points_mask], device=torch.device("cpu"))

                elif (self.params.rasterizer_mode is 'mesh'):
                    meshes = generate_tri_mesh(gt_models[i][0][points_mask], gt_models[i][1][points_mask], self.params.mesh_radius, gt_models[i][2][points_mask],
                                               device=torch.device("cpu"))
            else:
                if (self.params.rasterizer_mode is 'pointcloud'):
                    point_clouds = generate_pointcloud(gt_models[i][0], gt_models[i][1],
                                                       gt_models[i][2], device = torch.device("cpu"))

                elif (self.params.rasterizer_mode is 'mesh'):
                    meshes = generate_tri_mesh(gt_models[i][0], gt_models[i][1], self.params.mesh_radius, gt_models[i][2], device = torch.device("cpu"))

            if self.params.mask_render:
                slice_total_mask = np.load('/home/jinzhi//hdd10T/aMyLab/project_render/record/zhiwei/edit_ply/msak.npy')
                idx = np.where(slice_total_mask == 7)
                slice_mask = np.zeros(slice_total_mask.shape, dtype = np.bool_)
                slice_mask[idx] = True

                if (self.params.rasterizer_mode is 'pointcloud'):
                    result_mask = np.zeros(point_clouds.points_padded().shape[1], dtype = np.bool_)
                elif (self.params.rasterizer_mode is 'mesh'):
                    result_mask = np.zeros(meshes.faces_padded().shape[1], dtype = np.bool_)
                count = -1
                for i in range(points_mask.shape[0]):
                    if points_mask[i] == True:
                        count += 1
                        if slice_mask[i] == True:
                            result_mask[count] = True

                self.slice_mask = result_mask
            else:
                if (self.params.rasterizer_mode is 'pointcloud'):
                    self.slice_mask = np.ones(point_clouds.points_padded().shape[1], dtype = np.bool_)
                elif (self.params.rasterizer_mode is 'mesh'):
                    self.slice_mask = np.ones(meshes.faces_padded().shape[1], dtype = np.bool_)

            # self.meshes_list.append(meshes)
            # self.basic_len.append(meshes.faces_padded().shape[1] + 1)

            if (self.params.rasterizer_mode is 'pointcloud'):
                self.pointclouds_list.append(point_clouds)
                self.basic_len.append(point_clouds.points_padded().shape[1] + 1)
                z_embeddings = torch.nn.Embedding((point_clouds.points_padded().shape[1] + 1) * sum_num,
                                                  self.params.z_length, max_norm=1,
                                                  sparse=self.params.use_sparse_embedding)

            elif (self.params.rasterizer_mode is 'mesh'):
                self.meshes_list.append(meshes)
                self.basic_len.append(meshes.faces_padded().shape[1] + 1)
                z_embeddings = torch.nn.Embedding((meshes.faces_padded().shape[1] + 1) * sum_num,
                                                  self.params.z_length, max_norm=1,
                                                  sparse=self.params.use_sparse_embedding)

            # z_embeddings = torch.nn.Embedding((meshes.faces_padded().shape[1] + 1) * sum_num,
            #                                   self.params.z_length, max_norm=1, sparse = self.params.use_sparse_embedding)
            # # z_embeddings = torch.nn.Embedding((meshes.faces_padded().shape[1] + 1) * self.params.embedding_level,
            #                                   self.params.z_length, max_norm=1)
            torch.nn.init.normal_(z_embeddings.weight.data, 0.0, std=0.1)
            self.z_embeddings_list.append(z_embeddings)
            if(self.params.use_lighting_embedding):
                lighting_embedding = torch.nn.Embedding(len(self.params.light_condition_list), self.params.embedding_light_length, max_norm=1, sparse = self.params.use_sparse_embedding)
                torch.nn.init.normal_(lighting_embedding.weight.data, 0.0, std=0.1)
                self.lighting_embedding_list.append(lighting_embedding)

        print('finished initialization')

        if (self.params.backward_gradient_mode is 'meta'):

            self.optimizer_network = None
            self.optimizer_embeddings = None
            if (self.params.use_sparse_embedding):
                self.optimizer_network_meta = torch.optim.Adam(
                    [
                        {
                            "params": filter(lambda p: p.requires_grad, self.render_machine_meta.parameters()),
                            "lr": self.params.lr_net_2d_meta
                        },
                    ]
                )
            print('finished meta optimizer initialization')

    def load_checkpoint(self, filename):
        if(filename is None):
            print('do not load checkpoint')
        else:
            try:
                print("Loading checkpoint '{}'".format(filename))
                checkpoint = torch.load(filename)

                #self.current_epoch = checkpoint['epoch']
                #self.current_iteration = checkpoint['iteration']
                #self.lr = checkpoint['lr']
                if(self.params.backward_gradient_mode is 'meta'):
                    self.render_machine_meta.load_state_dict(checkpoint['render_machine_dict'])
                else:
                    self.render_machine.load_state_dict(checkpoint['render_machine_dict'])

                if(self.params.load_optimizer_network):
                    self.optimizer_network.load_state_dict(checkpoint['optimizer_network'])
                if(self.params.load_lighting_embedding_list):
                    self.lighting_embedding_list.load_state_dict(checkpoint['lighting_embedding_list'])

                    #pass
                if(self.params.load_z_embeddings_list):
                    count = 1
                    sum_num = 0
                    for i in range(self.params.load_stage):
                        sum_num += count
                        count = count * self.params.stage_base_num

                    z_embeddings_list0 = nn.ModuleList()
                    for i in range(len(self.params.modelList)):
                        z_embeddings0 = torch.nn.Embedding(self.basic_len[i] * sum_num,
                                                        self.params.z_length, max_norm=1)
                        z_embeddings_list0.append(z_embeddings0)

                    z_embeddings_list0.load_state_dict(checkpoint['z_embeddings_list'])

                    for i in range(len(self.params.modelList)):
                        embedding_ex1 = z_embeddings_list0[i].weight
                        embedding_last = z_embeddings_list0[i].weight[-self.basic_len[i] * int(count / self.params.stage_base_num):]

                        embedding_ex = embedding_ex1
                        count_temp = 1
                        for j in range(self.params.init_stage - self.params.load_stage):
                            count_temp = count_temp * self.params.stage_base_num
                            embedding_ex2 = embedding_last.unsqueeze(1).expand(embedding_last.shape[0],count_temp,embedding_last.shape[1]).reshape(-1,embedding_last.shape[1])
                            embedding_ex = torch.cat((embedding_ex,embedding_ex2),0)

                        self.z_embeddings_list[i].weight.data.copy_(embedding_ex)
                print("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                      .format(self.params.load_checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))


                # torch.nn.init.normal_(self.render_machine.render_net_c.conv_out_4.weight, mean=0.0, std=1e-3)
                # torch.nn.init.normal_(self.render_machine.render_net_c.conv_out_4.bias, mean=0.0, std=1e-3)
                # torch.nn.init.normal_(self.render_machine.render_net_l.conv_s_4.weight, mean=0.0, std=1e-3)
                # torch.nn.init.normal_(self.render_machine.render_net_l.conv_s_4.bias, mean=0.0, std=1e-3)
            except OSError as e:
                self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
                self.logger.info("**First time to train**")

    def save_checkpoint(self,  is_best=True):
        """

        """
        file_name = 'epoch:%s.pth.tar'%str(self.current_epoch).zfill(5)

        if(self.params.backward_gradient_mode is 'meta'):
            state = {
                'epoch': self.current_epoch,
                'lr': self.lr,
                'iteration': self.current_iteration,
                'render_machine_dict': self.render_machine_meta.state_dict(),
                'lighting_embedding_list':self.lighting_embedding_list.state_dict(),
                'z_embeddings_list':self.z_embeddings_list.state_dict(),
                'optimizer_network': self.optimizer_network_meta.state_dict(),
                'optimizer_embeddings': self.optimizer_embeddings.state_dict(),
            }
        else:
            state = {
                'epoch': self.current_epoch,
                'lr': self.lr,
                'iteration': self.current_iteration,
                'render_machine_dict': self.render_machine.state_dict(),
                'lighting_embedding_list': self.lighting_embedding_list.state_dict(),
                'z_embeddings_list': self.z_embeddings_list.state_dict(),
                'optimizer_network': self.optimizer_network.state_dict(),
                'optimizer_embeddings': self.optimizer_embeddings.state_dict(),
            }

        if not os.path.exists(os.path.join(self.params.checkpoint_dir, '')):
            os.makedirs(os.path.join(self.params.checkpoint_dir, ''))
        # Save the state
        torch.save(state, os.path.join(self.params.checkpoint_dir, file_name))
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(os.path.join(self.params.checkpoint_dir, file_name),
                            os.path.join(self.params.checkpoint_dir, 'best.pth.tar'))


    def run(self):
        """
        The main operator
        :return:
        """
        self.train()

    def update_settings(self, load_state = False):

        # if not(self.params.backward_gradient_mode is 'meta'):
        #     self.lr *= self.params.lr_decay_rate

        if not(self.params.use_D_prediction):
            for para in self.render_machine.shading_machine.render_net_g.parameters():
                para.requires_grad = False
        else:
            for para in self.render_machine.shading_machine.render_net_g.parameters():
                para.requires_grad = True

        if not(self.params.use_C_prediction):
            for para in self.render_machine.shading_machine.render_net_c.parameters():
                para.requires_grad = False
        else:
            for para in self.render_machine.shading_machine.render_net_c.parameters():
                para.requires_grad = True

        if(self.params.backward_gradient_mode is 'together'):
            if(self.params.use_sparse_embedding):

                self.optimizer_network = torch.optim.Adam(
                    [
                        {
                            "params": filter(lambda p: p.requires_grad, self.render_machine.parameters()),
                            "lr": self.params.lr_net_2d
                        },
                    ]
                )
                self.optimizer_embeddings = torch.optim.SparseAdam(
                    [
                        {
                            "params": self.z_embeddings_list.parameters(),
                            "lr": self.params.lr_embeddings,
                        },
                        {
                            "params": self.lighting_embedding_list.parameters(),
                            "lr": self.params.lr_embeddings
                        },
                    ]
                )

        elif (self.params.backward_gradient_mode is 'meta'):

            if(self.optimizer_network is not None):
                state_network = self.optimizer_network.state_dict()
            if (self.optimizer_embeddings is not None):
                state_embedding = self.optimizer_embeddings.state_dict()

            if (self.params.use_sparse_embedding):

                self.optimizer_network = torch.optim.Adam(
                    [
                        {
                            "params": filter(lambda p: p.requires_grad, self.render_machine.parameters()),
                            "lr": self.params.lr_net_2d
                        },
                    ]
                )
                self.optimizer_embeddings = torch.optim.SparseAdam(
                    [

                        {
                            "params": self.z_embeddings_list.parameters(),
                            "lr": self.params.lr_embeddings,
                        },
                        {
                            "params": self.lighting_embedding_list.parameters(),
                            "lr": self.params.lr_embeddings
                        },
                    ]
                )

            if load_state:
                self.optimizer_network.load_state_dict(state_network)
                self.optimizer_embeddings.load_state_dict(state_embedding)

            print('finished meta optimizer preparition')

    def train(self):

        print('start training')
        print('########################################################')
        Dataset = DataLoader(self.MVSDataset, batch_size=1, shuffle=False, num_workers=12)
        Dataset_test = DataLoader(self.MVSDataset_test, batch_size=1, shuffle=False, num_workers=12)
        for epoch in range(self.current_epoch, self.params.max_epoch):

            self.current_epoch = epoch
            print('start training %s epoch' % str(self.current_epoch))

            if(epoch % self.params.save_checkpoint_iter == 0):
                self.save_checkpoint()
            if(epoch % self.params.change_params_iter == 0):
                light_condition = random.choice(self.params.light_condition_random_list)
                self.light_condition = int(light_condition)
                #self.render_machine.lighting_condition_id = self.light_condition
                if(self.params.backward_gradient_mode is 'meta'):
                    self.render_machine_meta.lighting_condition_id = 0
                else:
                    self.render_machine.lighting_condition_id = 0
                #resol = random.uniform(self.params.random_resol_min, self.params.random_resol_max)
                # change the parameters (lighting_condition, #resol)
                self.params.change_params(light_condition=light_condition)
                Dataset = DataLoader(self.MVSDataset, batch_size=1, shuffle=False, num_workers=12)
                if not (self.params.backward_gradient_mode is 'meta'):
                    self.update_settings()

            self.params.train_strategy(epoch)
            self.train_one_epoch(Dataset)

    def train_one_epoch(self, Dataset):

        for index_dataset, sample in enumerate(Dataset,0):
            print('index_dataset', index_dataset)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

            torch.cuda.empty_cache()
            N_group = len(sample['train']['image'])
            self.N_group = N_group

            if(self.params.backward_gradient_mode is 'meta'):
                self.render_machine = self.render_machine_meta.clone()

                # self.z_embedding = self.z_embeddings_list_meta[index_dataset].to(self.device).clone()
                # if (self.params.use_lighting_embedding):
                #     self.lighting_embedding = self.lighting_embedding_list_meta[index_dataset].to(self.device).clone()
                # else:
                #     self.lighting_embedding = None
                self.z_embedding = self.z_embeddings_list[index_dataset].to(self.device)
                if (self.params.use_lighting_embedding):
                    self.lighting_embedding = self.lighting_embedding_list[index_dataset].to(self.device)
                else:
                    self.lighting_embedding = None

                self.update_settings(load_state = True)

            else:
                self.z_embedding = self.z_embeddings_list[index_dataset].to(self.device)
                if (self.params.use_lighting_embedding):
                    self.lighting_embedding = self.lighting_embedding_list[index_dataset].to(self.device)
                else:
                    self.lighting_embedding = None

            for group_i in range(N_group):
                torch.cuda.empty_cache()
                for j in range(self.params.image_compress_stage):

                    if (self.params.rasterizer_mode is 'pointcloud'):
                        mesh = self.pointclouds_list[index_dataset]
                    elif (self.params.rasterizer_mode is 'mesh'):
                        mesh = self.meshes_list[index_dataset]

                    mesh = mesh.to(self.device).extend(sample['train']['meta'][group_i]['view_list'][0,:].shape[0])

                    parti_imgs = sample['train']['img_patches'][group_i][0, :]
                    _cameraP04s_models = sample['train']['camP0s_list'][group_i][0, :]

                    if (self.params.use_random_partition):
                        if (self.params.patch_random_num <= int(self.params.compress_ratio_h * self.params.compress_ratio_w)):
                            partition_random_list = np.random.choice(self.params.partition_list, self.params.patch_random_num, replace=False, p=None)
                        else:
                            partition_random_list = self.params.partition_list
                    else:
                        partition_random_list = self.params.partition_list

                    for partition_j in partition_random_list:
                        print('Dealing with partition: ', str(partition_j))
                        mask, image_out, zbuf = self.render_machine(mesh,
                                                                    _cameraP04s_models[:, partition_j, :, :].to(self.device),
                                                                    self.MVSDataset.cameraTs_new[index_dataset][sample['train']['meta'][group_i]['view_list'][0, :].numpy().tolist(), ...].to(self.device),
                                                                    self.MVSDataset.BB_models_tensor[index_dataset].to(self.device),
                                                                    self.params.image_size_single.to(self.device),
                                                                    self.z_embedding,
                                                                    (int(self.params.img_h / (self.params.compress_ratio_total * self.params.compress_ratio_h)),
                                                                     int(self.params.img_w / (self.params.compress_ratio_total * self.params.compress_ratio_w))),
                                                                    rasterize_type='train',
                                                                    camera_intrisic = self.camera_intrisic,
                                                                    lighting_embedding = self.lighting_embedding,
                                                                    )

                        loss_zbuf = (self.params.loss_feature_var_weight * self.render_machine.pixel_features_var + self.render_machine.pixel_zbufs_var)

                        loss = self.mse(mask * image_out, mask * mask * parti_imgs[:, partition_j, :, :, :].type(mask.dtype).to(self.device))

                        loss_total = self.loss_recorder([loss, loss_zbuf])

                        #print('loss_total', loss_total)
                        self.optimizer_network.zero_grad()
                        self.optimizer_embeddings.zero_grad()

                        loss_total.backward()

                        self.optimizer_network.step()
                        self.optimizer_embeddings.step()

                        if(self.current_epoch % self.params.draw_cubic_iter == 0) and (group_i is 0) and (partition_j == (int(self.params.compress_ratio_h * self.params.compress_ratio_w / 2))):
                            self.summary_writer.add_images('image predict stage0/input stage %s' % str(j), image_out.detach().cpu() + 0.5, self.current_iteration)

            if (self.params.backward_gradient_mode is 'meta'):
                self.render_machine_meta.point_grad_to(self.render_machine)
                #self.z_embeddings_list_meta.point_grad_to(self.z_embeddings_list)

                self.optimizer_network_meta.step()
                # self.optimizer_embeddings_meta.step()

                print('finished meta network update')

            avg_loss_list_out = self.loss_recorder.return_avg()
            self.summary_writer.add_scalar("epoch/loss", avg_loss_list_out[0], self.current_iteration)
            self.summary_writer.add_scalar("epoch/loss zbuf", avg_loss_list_out[1], self.current_iteration)

            if(self.current_epoch % self.params.draw_cubic_iter == 0):

                for i in range(self.params.image_compress_stage):
                    self.summary_writer.add_images('image gt/input stage %s'%str(i), sample['train']['image'][0][i][0] + 0.5, self.current_iteration)

            self.current_iteration += 1



    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError


    def refine_points(self):
        with torch.no_grad():
            #model_i = 0
            for model_i in range(len(self.params.modelList)):
                self.mesh = self.meshes_list[model_i]

                self.faces_index = torch.zeros(self.mesh.faces_packed().shape[0]).to(self.device)
                print("initial refinement points number: ", str(self.faces_index.shape[0]))
                #pdb.set_trace()
                #z_embedding = self.z_embeddings_list[model_i]

                file_root_reconstruction = os.path.join(self.params.root_file, self.params.renderResultRoot_preprocess,
                                            str(self.params.modelList[model_i]), 'points_filter')
                if not os.path.exists(file_root_reconstruction):
                    os.makedirs(file_root_reconstruction)

                #sample_resolution = self.params.sample_resolution
                # margin_init = self.params.margin_init
                # margin_end = self.params.margin_end
                # ply_filePath = os.path.join(file_root_reconstruction, "2sample:%s.ply" % (str(sample_resolution)))

                for group_i in range(int(len(self.params.testViewList)//self.params.group_pair_num_max)+1):

                    id_init = self.params.group_pair_num_max * group_i
                    id_end = self.params.group_pair_num_max * (group_i+1) if (group_i is not (int(len(self.params.testViewList)//self.params.group_pair_num_max))) else len(self.params.testViewList)
                    #pdb.set_trace()
                    if(id_init == id_end):
                        break
                    mesh = self.mesh.to(self.device).extend(id_end - id_init)

                    self.faces_index = self.render_machine.refine_points(
                                                                        mesh,
                                                                        self.MVSDataset_test.cameraPO4s_models[model_i][id_init:id_end].to(self.device),
                                                                        self.MVSDataset_test.cameraTs_new[model_i][id_init:id_end].to(self.device),
                                                                        self.MVSDataset_test.BB_models_tensor[model_i].to(self.device),
                                                                        self.params.image_size.to(self.device),
                                                                        self.faces_index
                                                                    )
                self.faces_index = (self.faces_index > 0)
                print("final refinement points number: ", str(self.faces_index.sum()))
                np.save(os.path.join(file_root_reconstruction, 'mask.npy'), self.faces_index.cpu().numpy()) # save shape: (H,W)
                print('end')
                #pdb.set_trace()


    def reconstruct(self):

        for model_i in range(len(self.params.modelList)):

            with torch.no_grad():
                if (self.params.rasterizer_mode is 'pointcloud'):
                    self.mesh = self.pointclouds_list[model_i]
                elif (self.params.rasterizer_mode is 'mesh'):
                    self.mesh = self.meshes_list[model_i]
                self.z_embeddings_list.to(self.device)
                if(self.params.use_lighting_embedding):
                    self.lighting_embedding_list.to(self.device)
                z_embedding = self.z_embeddings_list[model_i]
                if(self.params.use_lighting_embedding):
                    lighting_embedding = self.lighting_embedding_list[model_i]
                else:
                    lighting_embedding = None

                self.graph_reconstruct.create_file(model_i)

                if(self.params.return_points_colour):
                    points_coords_global_new, points_colour_global_new, points_normal_global_new = self.render_machine.points_generation(
                        self.mesh, z_embedding,
                        sample_resolution=self.graph_reconstruct.sample_resolution,
                        margin_init=self.graph_reconstruct.margin_init,
                        margin_end=self.graph_reconstruct.margin_end,
                        camera_position=self.MVSDataset_test.cameraTs_new[model_i][0:1],
                        lighting_embedding = lighting_embedding,
                        BB = self.MVSDataset_test.BB_models_tensor[model_i].to(self.device)
                        )
                    save2ply(self.graph_reconstruct.ply_filePath, points_coords_global_new.numpy(), rgb_np=points_colour_global_new.numpy(), normal_np=points_normal_global_new.numpy())
                else:
                    points_coords_global_new = self.render_machine.points_generation(self.mesh,
                                                                                     z_embedding,
                                                                                     sample_resolution = self.graph_reconstruct.sample_resolution,
                                                                                     margin_init=self.graph_reconstruct.margin_init,
                                                                                     margin_end=self.graph_reconstruct.margin_end,
                                                                                     camera_position = self.MVSDataset_test.cameraTs_new[model_i][0:1],
                                                                                     lighting_embedding = lighting_embedding,
                                                                                     BB = self.MVSDataset_test.BB_models_tensor[model_i].to(self.device)
                                                                                     )
                    save2ply(self.graph_reconstruct.ply_filePath, points_coords_global_new.numpy(), rgb_np=None, normal_np=None)

            torch.cuda.empty_cache()
            with torch.no_grad():

                self.image_visualizer.create_file(model_i)
                print('Create visualizer file finished !')
                for group_i in range(int(len(self.params.testViewList)//self.params.group_pair_num_max)+1):
                    torch.cuda.empty_cache()
                    id_init = self.params.group_pair_num_max * group_i
                    id_end = self.params.group_pair_num_max * (group_i+1) if (group_i is not (int(len(self.params.testViewList)//self.params.group_pair_num_max))) else len(self.params.testViewList)
                    if(id_init == id_end):
                        break

                    mesh = self.mesh.to(self.device).extend(id_end - id_init)

                    _cameraP04s_models = self.MVSDataset_test.camP0s_models[model_i][id_init:id_end]
                    parti_imgs = self.MVSDataset_test.img_patches_models[model_i][id_init:id_end]


                    self.image_visualizer.initialize_empty_images(id_end - id_init)

                    for partition_j in range(int(self.params.compress_ratio_h * self.params.compress_ratio_w)):

                        mask, images, zbuf = self.render_machine(
                                                                mesh,
                                                                # self.MVSDataset_test.camP0s_models[model_i][id_init:id_end][:, partition_j, :,:].to(self.device),
                                                                _cameraP04s_models[:, partition_j, :,:].to(self.device),
                                                                self.MVSDataset_test.cameraTs_new[model_i][id_init:id_end].to(self.device),
                                                                self.MVSDataset_test.BB_models_tensor[model_i].to(self.device),
                                                                self.params.image_size_single.to(self.device),
                                                                z_embedding,
                                                                (int(self.params.img_h/(self.params.compress_ratio_total * self.params.compress_ratio_h)),int(self.params.img_w/(self.params.compress_ratio_total * self.params.compress_ratio_w))),
                                                                rasterize_type='test',
                                                                camera_intrisic = self.camera_intrisic_test,
                                                                lighting_embedding = lighting_embedding,
                                                                slice_mask = self.slice_mask
                                                                )
                        images = torch.clamp(images * mask, -0.5, 0.5)
                        self.image_visualizer.combine_images(partition_j, mask, images, zbuf, id_end, id_init, parti_imgs)
                    self.image_visualizer.save_total_image(id_end, id_init)


    def render_novel_view(self, return_stage = 0, lighting_condition_id = None):

        if(lighting_condition_id is not None):
            self.render_machine.lighting_condition_id = lighting_condition_id
        with torch.no_grad():
            model_i = 0
            for model_i in range(len(self.params.modelList)):
                if (self.params.rasterizer_mode is 'pointcloud'):
                    self.mesh = self.pointclouds_list[model_i]
                elif (self.params.rasterizer_mode is 'mesh'):
                    self.mesh = self.meshes_list[model_i]

                z_embedding = self.z_embeddings_list[model_i].to(self.device)
                lighting_embedding = self.lighting_embedding_list[model_i].to(self.device)

                self.image_visualizer.create_file(model_i)

                _cameraP04s_model = self.MVSDataset_test.camP0s_models_interpolate[model_i]
                num_view_interpolate = _cameraP04s_model.shape[0]

                print(self.params.interpolate_novel_view_num)
                print(_cameraP04s_model.shape)

                self.image_visualizer.initialize_empty_images(num_view_interpolate)

                for group_i in range(int(num_view_interpolate // self.params.group_pair_num_max) + 1):
                    torch.cuda.empty_cache()
                    id_init = self.params.group_pair_num_max * group_i
                    id_end = self.params.group_pair_num_max * (group_i + 1) if (
                                group_i is not (int(num_view_interpolate // self.params.group_pair_num_max))) else int(num_view_interpolate)
                    if (id_init == id_end):
                        break
                    mesh = self.mesh.to(self.device).extend(id_end - id_init)


                    for partition_j in range(int(self.params.compress_ratio_h * self.params.compress_ratio_w)):
                        mask, images, zbuf = self.render_machine(
                                                                mesh,
                                                                # self.MVSDataset_test.camP0s_models_interpolate[model_i][id_init:id_end, partition_j, :, :].to(self.device),
                                                                _cameraP04s_model[id_init:id_end, partition_j, :, :].to(self.device),
                                                                self.MVSDataset_test.cameraTs_interpolate[model_i][id_init:id_end].to(self.device),
                                                                self.MVSDataset_test.BB_models_tensor[model_i].to(self.device),
                                                                self.params.image_size_single.to(self.device),
                                                                z_embedding,
                                                                (int(self.params.img_h / (self.params.compress_ratio_total * self.params.compress_ratio_h)),
                                                                 int(self.params.img_w / (self.params.compress_ratio_total * self.params.compress_ratio_w))),
                                                                rasterize_type='test',
                                                                camera_intrisic = self.camera_intrisic_test,
                                                                lighting_embedding = lighting_embedding)

                        images = torch.clamp(images * mask, -0.5, 0.5)

                        self.image_visualizer.combine_images(partition_j, mask, images, zbuf, id_end, id_init, None)

                    self.image_visualizer.save_total_image(id_end, id_init)

                self.image_visualizer.generate_gif(num_view_interpolate)



if __name__ == '__main__':
    params = Params()
    params.change_render_mode()
    agent = BaseAgent(params)



    if(params.mode is 'reconstruct'):
        print('start reconstruct #@~#@#@#!@#@!#')
        agent.reconstruct()
    elif(params.mode is 'refine_points'):
        print('start refine_points #@~#@#@#!@#@!#')
        agent.refine_points()
    elif(params.mode is 'render_novel_view'):
        print('start render_novel_view #@~#@#@#!@#@!#')
        agent.render_novel_view(return_stage = 0, lighting_condition_id = 0)
    elif(params.mode is 'train'):
        print('start train #@~#@#@#!@#@!#')
        agent.train()
