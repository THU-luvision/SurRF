import numpy as np
import os
import torch
import math


class Params(object):

    def __init__(self):
        self.exp_id = '0207/exp1'
        self.root_params()
        self.network_params()
        self.train_params()
        self.load_params()
        self.reconstruct_params()

    def change_node_num(self, node_num):
        self.node_num = node_num

    def change_params(self, light_condition, resol=None):
        self.light_condition = light_condition
        # self.resol = resol
        print('change the parameter successfully')
        print('self.light_condition', self.light_condition)

    def root_params(self):
        self._datasetName = 'DTU'
        # specify the root folder to save experiment results.
        self.root_file = ''
        # specify the root folder of dataset.
        self._input_data_rootFld = ''
        # specify the directory for trained ckpt.
        self.load_checkpoint_dir = None
        # specify the running mode.
        self.mode = 'train'  # reconstruct/render_novel_view/train

        self._debug_data_rootFld = os.path.join(self.root_file, 'debug', self.exp_id)
        self.summary_dir = os.path.join(self.root_file, 'experiment/train/log/', self.exp_id)
        print('self.summary_dir', self.summary_dir)
        self.checkpoint_dir = os.path.join(self.root_file, 'experiment/train/state', self.exp_id)
        self.rasterizer_mode = 'mesh'
        self.load_z_embeddings_list = True
        self.load_lighting_embedding_list = True
        self.load_optimizer_network = False

        self.node_num = [1]  # [1,4,16]/[1]
        self.init_stage = 1
        self.load_stage = 1
        self.stage_base_num = 4
        self.stage_block = [0]  # [0,10,40]
        self.render_ob_flag = 0
        self.inter_zoomin = False
        self.inter_choose = [0, 1, 0, 1]
        self.zoomin_rate = [0, 400, 0, -100]
        self.mask_render = False
        self.edit_flag = False
        self.save_mode = 'alpha'  # gray/white/alpha
        self.draw_cubic_dir = os.path.join(self.root_file, 'experiment/train/cubic', self.exp_id)

    def network_params(self):
        self.image_compress_stage = 1  # this indicate the compress stage of network
        self.image_compress_multiple = int(2 ** (self.image_compress_stage - 1))

        self.descriptor_length = 48  # 32
        self.descriptor_length_c = 48  # 32 12

        self.z_length_d = 48  # 32
        self.z_length_c = 48
        self.z_length = self.z_length_d + self.z_length_c

        self.descriptor_light_length = 8  # 8/4
        self.embedding_light_length = 4  # 4/2

        self.use_verts_grad = False
        self.use_2d_network = False

        self.use_feature_alpha = False
        self.use_lighting_mlp = True
        self.use_lighting_embedding = True
        self.use_D_prediction = True
        self.use_C_prediction = True

        self.use_relative_resolution = False
        self.use_random_ralative_resolution_ratio = False
        self.relative_resolution_normalize_ratio = 1e3
        self.render_image_size_normalize_ratio = 300

    def train_params(self):
        self.use_cuda = True
        if (self.use_cuda and torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.max_epoch = 100000

        self.lr_net_2d = 2e-3
        self.lr_embeddings = 2e-3

        self.lr_net_2d_meta = 0.2
        self.lr_embeddings_meta = 0.2

        self.draw_cubic_iter = 500  # 10/200
        self.save_checkpoint_iter = 200  # 20/500
        self.change_params_iter = 1000000
        self.validate_iter = 1000000

        self.loss_img_weight = 10.0
        self.loss_zbuf_weight = 30.0
        # self.loss_zbuf_weight = 0.0
        self.loss_zbuf_residual_weight = 1e-1
        self.loss_avg_weight = 0.1
        self.loss_feature_var_weight = 1e0

        self.backward_gradient_mode = 'together'  # 'together'/'meta'

    def train_strategy(self, epoch):
        pass

    def reconstruct_params(self):
        self.total_gpu_memory = 1e8
        self.return_points_colour = True
        self.use_points_blast = False
        # specify the resolution for the point cloud reconstruction.
        self.sample_resolution = 6
        self.margin_init = 0.0
        self.margin_end = 1.0

        self.lighting_predict_mode = 'total'  # total/s/z/n

    def load_params(self):

        self.load_view_selection_file = False
        self.use_sparse_embedding = True

        if (self._datasetName is 'DTU'):
            self.resol_gt = 0.2
            # the coarse input point cloud resolution.
            self.resol_gt_render = 4.0
            # specify the rendering configuration.
            self.render_image_size = 400  # the rendered output size, e.g. 400 means 400x400.
            self.random_render_image_size = [360, 400]  # random rendered size for training.
            self.compress_ratio_h = 1  # partition num along the h axis.
            self.compress_ratio_w = 1  # partition num along the w axis.
            self.compress_ratio_total = 4  # downsample ratio for input images.

            self.datasetFolder = os.path.join(self._input_data_rootFld, 'DTU_MVS')
            self.imgNamePattern = "Rectified/scan$/rect_#_&_r5000.{}".format('png')
            self.poseNamePattern = "SampleSet/MVS Data/Calibration/cams/00000#_cam.txt"  # replace # to {:03} SampleSet/MVS Data/Calibration/cal18/pos_#.txt

            self.BBNamePattern = "SampleSet/MVS Data/ObsMask/ObsMask#_10.mat"
            self.tranMatPattern = "advanced/Courthouse/Courthouse_trans.txt"
            self.gtNamePattern = 'Points/stl/stl#_total.ply'
            self.renderGtNamePattern = 'preprocess/ply_rmvs/#/$_surface_xyz_flow.ply'.replace('$',
                                                                                              str(self.resol_gt_render))
            self.renderResultRoot = 'experiment/render/image/view/resol:%s' % str(self.resol_gt_render)
            self.renderResultRoot_preprocess = 'experiment/render/numpy/id:20200608/resol:%s' % str(
                self.resol_gt_render)
            self.save_depth = False
            self.load_zbuf = False

            self.count_normal = False

            self.quantiGTPattern = 'Dataset/model_id:#/id:&_/'

            self.blur_radius = 2e-6
            self.faces_per_pixel = 3
            self.zbuf_front_threshold = 4.0
            self.mesh_radius = self.resol_gt_render * 1.2

            self.D_length = self.resol_gt_render * 1.0
            self.D_min_length = 2.0
            self.noise_coords_length = 1.0

            self.siren_omega = 1  # 30

            self.load_preprocess_index = False

            self.light_condition = '3'
            self.light_condition_list = ['3']  # ['0','1','2','3','4','5','6']
            self.light_condition_random_list = ['3']  # ['1','2','3','4','5']
            self.modelList = [9]
            # self.modelList = [1,4,15,23,114]
            # self.modelList = [1,4,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118]
            # self.modelList = [1,4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118]

            self.testViewList = [22,23,25,26]
            # self.testViewList = [25]
            # self.testViewList = [13,14,15,16,24,25,26,27,31,32,33,34]
            # self.testViewList = [22,23,24,25,33,34,35,36,37,42,43,44,45] #35 is the center
            # self.testViewList = range(1,49)

            # self.trainViewList = [25]
            # self.trainViewList = range(1, 49, 2)
            self.trainViewList = [13, 14, 15, 16, 24, 25, 26, 27, 31, 32, 33, 34]
            # self.renderViewList = range(1,49,3)

            self.interpolate_novel_view_num = 3
            self.interpolate_direction = -1  # 1,-1

            self.outer_bound_factor = 0.0

            self._cube_D_ = 1

            self.relative_resolution_network_random_ratio = [0.8 ** 2, 6.2 ** 2]

            self.relative_resolution_normalize_ratio = 1e3
            self.render_image_size_normalize_ratio = self.render_image_size

            # the original size for the loaded input images.
            self.img_h = 1200
            self.img_w = 1600

            self.image_size = torch.FloatTensor([[[self.img_w, self.img_h]]])  # the size of the input image
            self.image_size_single = torch.FloatTensor([[[int(
                self.img_w / (self.compress_ratio_w * self.compress_ratio_total)), int(
                self.img_h / (self.compress_ratio_h * self.compress_ratio_total))]]])  # the size of the input image

            self.use_random_partition = False
            self.partition_list = [patch for patch in range(int(self.compress_ratio_w * self.compress_ratio_h))]
            self.patch_random_num = 6  # the total num of patches for training, (< compress_ratio ** 2).

            self.estimate_normal = False

            self.resol = 0.8
            self.random_resol_min = self.resol
            self.random_resol_max = self.resol

            self.view_group_center = [0, 2, 5, 7, 9, 11]  # the center view number of each group pair
            self.in_group_angle = 360 * 3.1415926535 / 180  # the angle of pairs in a group
            self.group_pair_num_max = 4  # the maximum view number of each pair(usually it is 5)
            self.group_pair_num_min = 4  # the minimum view number of each pair(usually it is 2)
            self.min_group_num = 5  # the minimum group number
            self.max_group_num = 5  # the max group number

            self.augment_group_num = 0
            self.sample_alpha = 0.5
            self.sample_beta = 0.5
            self.sample_delta_alpha = 0.9
            self.sample_delta_beta = 0.9

            self.delta_length = 20.0

            self.rand_angle = 90

            self.z_axis_transform_rate = 1.0
            self.zbuf_min = 40
            self.zbuf_max = 80
            self.BB_shift = 200.0

    #     elif (self._datasetName is 'tanks_COLMAP'):
    #         self.resol_gt = 0.1
    #         self.resol_gt_render = 0.001
    #         self.resol_gt_finer = 1.0
    #
    #         self.datasetFolder = os.path.join(self._input_data_rootFld, 'tanks')
    #
    #         self.imgNamePattern = "intermediate/$/images/00000#.{}".format('jpg')
    #         self.poseNamePattern = "intermediate/$/cams/00000#_cam.txt"
    #         self.tranMatPattern = "advanced/Courthouse/Courthouse_trans.txt"
    #
    #         self.BBNamePattern = "intermediate/#/BB/bbox.npy"
    #         self.gtNamePattern = 'preprocess/intermediate/#/$_#.ply'
    #         self.renderGtNamePattern = 'preprocess/intermediate/#/$_#.ply'.replace('$', str(self.resol_gt_render))
    #         self.pairPattern = "intermediate/$/pair_mvsnet.txt"
    #         self.renderResultRoot = 'experiment/render/image/view/resol:%s' % str(self.resol_gt_render)
    #         self.renderResultRoot_numpy = 'experiment/render/numpy/id:20200608/resol:%s' % str(self.resol_gt_finer)
    #         self.renderResultRoot_preprocess = 'experiment/render/numpy/id:20200608/resol:%s' % str(
    #             self.resol_gt_render)
    #         self.save_depth = False
    #         self.load_zbuf = False
    #         self.load_mask = False
    #
    #         self.count_normal = False
    #
    #         self.quantiGTPattern = 'Dataset/model_id:#/id:&_/'
    #
    #         # self.image_size = torch.FloatTensor([[[2048,1080]]]) #Lighthouse Panther M60
    #         self.image_size = torch.FloatTensor([[[1920, 1080]]])  # Family Train Horse
    #
    #         # self.image_size_single = torch.FloatTensor([[[480, 270]]])  # the size of the input image
    #         self.render_image_size = 500  # the rendered output size
    #         self.random_render_image_size = [480, 500]
    #         self.blur_radius = 4e-6  # 1e-6
    #         self.faces_per_pixel = 3
    #         self.zbuf_front_threshold = 0.02  # Lighthouse:0.0005
    #         self.mesh_radius = self.resol_gt_render * 1.25
    #         self.implicit_texture_D = 0.025
    #         self.D_length = self.resol_gt_render * 3.0
    #         self.clamp_length = 0.02
    #         self.clamp_length_residual = 0.01
    #         self.D_min_length = 0.01
    #         self.noise_coords_length = 0.003
    #
    #         self.alpha_density_ratio = 10
    #         self.alpha_density_threshold = 0.1
    #         self.siren_omega = 30
    #
    #         self.preprocess_ahead_faces_num = 2
    #         self.preprocess_face_hit_min_num = 10  # 3 / 10
    #         self.load_preprocess_index = False
    #
    #         self.gamma_blend = 1e-4
    #         self.sigma_blend = 1e-4
    #
    #         self.light_condition = '3'
    #         self.light_condition_list = ['3']  # ['0','1','2','3','4','5','6']
    #         self.light_condition_random_list = ['3']  # ['1','2','3','4','5']
    #
    #         self.modelList = ['Horse']
    #
    #         self.relative_resolution_network_random_ratio = [0.8 ** 2, 4 ** 2]
    #
    #         self.relative_resolution_normalize_ratio = 1e3
    #         self.render_image_size_normalize_ratio = self.render_image_size
    #
    #         # self.testViewList = range(80,122,2)
    #         self.testViewList = [6]
    #
    #         # self.trainViewList = range(0,152,1)#range(0,151,1) horse/range(0,314,1) panther/range(0,313,1) m60/range(0,301,1) train
    #         # self.trainViewList = [6,14,15,23,26,30,34,36,40,47,49,63,67,74,77,85,88,90,94,103,112,113,120,123,130,132,135,140,142,148,151,155,163,166,170,175,231,244,247,255,265,267,273,306] #panther
    #         # self.trainViewList = [2,8,12,13,16,21,25,27,28,30,34,37,41,47,49,51,58,59,62,66,64,69,76,77,79,84,86,88,92,94,96,99,107,116,120,122,124,126,128,130,134,137,151,158,160,164,169,172,174,186,198,200,216,223,226,239,249,254,259,266,268,270,276,279,283,290,303] #m60
    #         # self.trainViewList = range(0,80,2) #Family
    #         # self.trainViewList = range(54,113,1) #Horse
    #
    #         self.trainViewList = range(0, 151, 1)
    #         self.interpolate_novel_view_num = 6
    #         self.interpolate_direction = -1  # 1,-1
    #
    #         self.outer_bound_factor = 0.0
    #
    #         self._cube_D_ = 1
    #         self.center_crop_ratio = 12.0 / 16.0
    #         self.cube_overlapping_ratio_for_training_sample = 0.5
    #
    #         self.compress_ratio_h = 1
    #         self.compress_ratio_w = 2
    #         self.compress_ratio_total = 2
    #
    #         self.img_h = 1080
    #         self.img_w = 1920  # Family Train Horse Francis
    #         # self.img_w = 2048 #Lighthouse Panther M60
    #
    #         ##############################

    #         self.image_size = torch.FloatTensor([[[self.img_w, self.img_h]]])  # the size of the input image
    #         # self.image_size_single = torch.FloatTensor([[[int(self.img_w / self.compress_ratio_total), int(self.img_h / self.compress_ratio_total)]]])  # the size of the input image
    #         self.image_size_single = torch.FloatTensor([[[int(
    #             self.img_w / (self.compress_ratio_w * self.compress_ratio_total)), int(
    #             self.img_h / (self.compress_ratio_h * self.compress_ratio_total))]]])  # the size of the input image
    #
    #         self.use_random_partition = False
    #         self.partition_list = [patch for patch in range(int(self.compress_ratio_w * self.compress_ratio_h))]
    #         self.patch_random_num = 8  # the total num of patches for training, (< compress_ratio ** 2).
    #         #############################
    #
    #         self.estimate_normal = False
    #
    #         self.resol = 0.8
    #         self.random_resol_min = self.resol
    #         self.random_resol_max = self.resol
    #
    #         self.load_view_selection_file = True
    #         # self.view_group_center = [0, 2, 5, 7, 10, 13, 16, 19,22, 25,28,31,34,37, 41]  #the center view number of each group pair
    #         # self.view_group_center = [0, 2, 5, 7, 10]  # the center view number of each group pair
    #         self.view_group_center = range(1, 58, 2)
    #         # self.view_group_center = [0]
    #         # self.view_group_center = [4, 13, 22, 31, 40, 47]  # the center view number of each group pair
    #         # self.group_pair_index = [[0,2],[1,3],[2,4],[3,5],[4,6],[5,7]]  #the index of the selected group
    #         # self.group_pair_index = [[0, 2], [1, 3], [3, 5], [4, 6], [0, 3], [4, 7],
    #         #                         [2, 5], ]  # the index of the selected group
    #         self.in_group_angle = 90 * 3.1415926535 / 180  # the angle of pairs in a group
    #         self.group_pair_num_max = 4  # the maximum view number of each pair(usually it is 5)
    #         self.group_pair_num_min = 4  # the minimum view number of each pair(usually it is 2)
    #         self.min_group_num = 6  # the minimum group number
    #         self.max_group_num = 6  # the max group number
    #
    #         self.augment_group_num = 0
    #         self.sample_alpha = 0.5
    #         self.sample_beta = 0.5
    #         self.sample_delta_alpha = 0.9
    #         self.sample_delta_beta = 0.9
    #
    #         self.delta_length = 20.0
    #
    #         self.rand_angle = 30
    #         self.rand_length = 2 / 1600.0  # 2/1600.0
    #         self.rand_K = 2  # 2
    #
    #         self.z_axis_transform_rate = 1e-2
    #         self.zbuf_min = 0
    #         self.zbuf_max = 100
    #
    #         self.BB_shift = 1.0
    #
    #         self.use_sparse_embedding = True
    #
    #     elif (self._datasetName is 'blendedMVS'):
    #         self.resol_gt = 2.0
    #         self.resol_gt_render = 0.4  # village
    #         self.resol_gt_finer = 1.0
    #
    #         self.datasetFolder = os.path.join(self._input_data_rootFld, 'blendedMVS')
    #         self.imgNamePattern = "$/highresol_images/00000#.{}".format('jpg')
    #         # self.imgNamePattern = "$/blended_images/00000#.{}".format('jpg')
    #         self.poseNamePattern = "$/highres_cams/00000#_cam.txt"
    #         # self.poseNamePattern = "$/cams/00000#_cam.txt"
    #         self.tranMatPattern = "advanced/Courthouse/Courthouse_trans.txt"
    #
    #         self.BBNamePattern = "#/BB/bbox.npy"
    #         self.gtNamePattern = '#/preprocess/ply/2.0_model.ply'
    #         self.renderGtNamePattern = '#/preprocess/ply/$_#.ply'.replace('$', str(self.resol_gt_render))
    #         self.pairPattern = "$/cams/pair.txt"
    #         self.renderResultRoot = 'experiment/render/image/view/resol:%s' % str(self.resol_gt_render)
    #         self.renderResultRoot_numpy = 'experiment/render/numpy/id:20200608/resol:%s' % str(self.resol_gt_finer)
    #         self.renderResultRoot_preprocess = 'experiment/render/numpy/id:20200608/resol:%s' % str(
    #             self.resol_gt_render)
    #         self.save_depth = False
    #         self.load_zbuf = False
    #         self.load_mask = False
    #
    #         self.count_normal = False
    #
    #         self.quantiGTPattern = 'Dataset/model_id:#/id:&_/'
    #
    #         self.compress_ratio_h = 2
    #         self.compress_ratio_w = 2
    #         self.compress_ratio_total = 1.5
    #
    #         self.img_h = 1536  # 1536 #576
    #         self.img_w = 2048  # 2048 #768
    #
    #         ##############################

    #         self.image_size = torch.FloatTensor([[[self.img_w, self.img_h]]])  # the size of the input image
    #         # self.image_size_single = torch.FloatTensor([[[int(self.img_w / self.compress_ratio_total), int(self.img_h / self.compress_ratio_total)]]])  # the size of the input image
    #         self.image_size_single = torch.FloatTensor([[[int(
    #             self.img_w / (self.compress_ratio_w * self.compress_ratio_total)), int(
    #             self.img_h / (self.compress_ratio_h * self.compress_ratio_total))]]])  # the size of the input image
    #
    #         self.use_random_partition = False
    #         self.partition_list = [patch for patch in range(int(self.compress_ratio_w * self.compress_ratio_h))]
    #         self.patch_random_num = 3  # the total num of patches for training, (< compress_ratio ** 2).
    #         #############################
    #
    #         self.render_image_size = 500  # the rendered output size
    #         self.random_render_image_size = [480, 500]
    #         self.blur_radius = 1e-6
    #         self.faces_per_pixel = 3
    #         self.zbuf_front_threshold = 0.05
    #         self.mesh_radius = self.resol_gt_render * 1.25
    #         self.implicit_texture_D = 0.025
    #         self.D_length = self.resol_gt_render * 3.0
    #         self.clamp_length = 3.0
    #         self.clamp_length_residual = 5.0
    #         self.D_min_length = 2.0
    #         self.noise_coords_length = 1.0
    #
    #         self.relative_resolution_normalize_ratio = 1e3
    #         self.render_image_size_normalize_ratio = self.render_image_size
    #
    #         self.alpha_density_ratio = 10
    #         self.alpha_density_threshold = 0.1
    #         self.siren_omega = 30
    #
    #         self.preprocess_ahead_faces_num = 2
    #         self.preprocess_face_hit_min_num = 10  # 3 / 10
    #         self.load_preprocess_index = False
    #
    #         self.gamma_blend = 1e-4
    #         self.sigma_blend = 1e-4
    #
    #         self.relative_resolution_network_random_ratio = [0.8 ** 2, 3.0 ** 2]
    #
    #         self.light_condition = '3'
    #         self.light_condition_list = ['3']  # ['0','1','2','3','4','5','6']
    #         self.light_condition_random_list = ['3']  # ['1','2','3','4','5']
    #
    #         # self.modelList = ['building','building3','building7','building8','building11','stadium2','stadium3','village3','village4']
    #         self.modelList = ['building']
    #
    #         ###############################

    #         # self.modelList = ['building8']
    #         # self.modelList = ['building3']
    #         # self.modelList = ['village3']
    #
    #         # stadium3
    #         self.testViewList = [4]  # [38]
    #         #################################

    #         # stadium2
    #         # self.testViewList = [7,72,88]
    #         # building
    #         # self.testViewList = [7,27,57]
    #         # building7
    #         # self.testViewList = []
    #         # building8
    #         # self.testViewList = [0,41,73]
    #         # building3
    #         # self.testViewList = [3,22,48]
    #         # building11
    #         # self.testViewList = [6,55,117]
    #         # village3
    #         # self.testViewList = [1,17,23]
    #         # village4
    #         # self.testViewList = [20,61,130]
    #
    #         # self.trainViewList = [4,6,7,8,12,17,18,21,26,30,35,36,38,39,40,49,56,59,67,76]
    #         # self.trainViewList = range(0, 186, 1)
    #         # stadium3
    #         # self.trainViewList = range(1, 78, 2)
    #         #####################################
    #         # # building
    #         # self.trainViewList = range(0, 96, 1)
    #         # # building7
    #         # self.trainViewList = range(0, 136, 2)
    #         # # building8
    #         # self.trainViewList = range(0, 77, 1) #range(1,75,2)
    #         # # building3
    #         # self.trainViewList = range(0, 68, 2)
    #         # # gaoida2
    #         # self.trainViewList = range(0, 89, 2)
    #         # # stadium2
    #         # self.trainViewList = range(0, 148, 2)
    #         # # village3
    #         # self.trainViewList = range(0, 74, 2) #range(1,74,2)
    #         # # village4
    #         self.trainViewList = range(0, 186, 1)
    #
    #         self.interpolate_novel_view_num = 6
    #         self.interpolate_direction = -1  # 1,-1
    #
    #         self.outer_bound_factor = 0.0
    #
    #         self._cube_D_ = 1
    #         self.center_crop_ratio = 12.0 / 16.0
    #         self.cube_overlapping_ratio_for_training_sample = 0.5
    #
    #         self.estimate_normal = False
    #
    #         self.resol = 0.8
    #         self.random_resol_min = self.resol
    #         self.random_resol_max = self.resol
    #
    #         # self.view_group_center = [0, 2, 5, 7, 10, 13, 16, 19,22, 25,28,31,34,37, 41]  #the center view number of each group pair
    #         # self.view_group_center = [0, 2, 5, 7, 10]  # the center view number of each group pair
    #         self.view_group_center = range(1, 94, 3)
    #         # self.view_group_center = [4, 13, 22, 31, 40, 47]  # the center view number of each group pair
    #         # self.group_pair_index = [[0,2],[1,3],[2,4],[3,5],[4,6],[5,7]]  #the index of the selected group
    #         # self.group_pair_index = [[0, 2], [1, 3], [3, 5], [4, 6], [0, 3], [4, 7],
    #         #                         [2, 5], ]  # the index of the selected group
    #         self.in_group_angle = 360 * 3.1415926535 / 180  # the angle of pairs in a group
    #         self.group_pair_num_max = 6  # the maximum view number of each pair(usually it is 5)
    #         self.group_pair_num_min = 6  # the minimum view number of each pair(usually it is 2)
    #         self.min_group_num = 6  # the minimum group number
    #         self.max_group_num = 6  # the max group number
    #
    #         self.augment_group_num = 0
    #         self.sample_alpha = 0.5
    #         self.sample_beta = 0.5
    #         self.sample_delta_alpha = 0.9
    #         self.sample_delta_beta = 0.9
    #
    #         self.delta_length = 20.0
    #
    #         self.rand_angle = 30
    #         self.rand_length = 2 / 1600.0  # 2/1600.0
    #         self.rand_K = 2  # 2
    #
    #         self.z_axis_transform_rate = 0.5
    #         self.zbuf_min = 40
    #         self.zbuf_max = 80
    #
    #         self.BB_shift = 150.0
    #
    #         self.use_sparse_embedding = True
    #
    #         self.load_view_selection_file = True
    #
    #     elif (self._datasetName is 'giga_ours'):
    #         self.resol_gt = 2.0
    #         self.resol_gt_render = 0.025
    #         self.resol_gt_finer = 1.0
    #
    #         self.datasetFolder = os.path.join(self._input_data_rootFld, 'giga_ours')
    #         self.imgNamePattern = "$/images/00000#.{}".format('jpg')
    #         self.poseNamePattern = "$/cams/00000#_cam.txt"
    #         self.tranMatPattern = "advanced/Courthouse/Courthouse_trans.txt"
    #
    #         self.BBNamePattern = "#/BB/bbox.npy"
    #         self.gtNamePattern = '#/preprocess/ply/2.0_model.ply'
    #         self.renderGtNamePattern = '#/preprocess/ply/$_surface_xyz_flow.ply'.replace('$', str(self.resol_gt_render))
    #         self.renderGtNamePattern4 = '#/preprocess/ply/$_surface_xyz_flow.ply'
    #         self.pairPattern = "$/pair.txt"
    #         self.renderResultRoot = 'experiment/render/image/view/resol:%s' % str(self.resol_gt_render)
    #         self.renderResultRoot_numpy = 'experiment/render/numpy/id:20200608/resol:%s' % str(self.resol_gt_finer)
    #         self.renderResultRoot_preprocess = 'experiment/render/numpy/id:20200608/resol:%s' % str(
    #             self.resol_gt_render)
    #         self.save_depth = False
    #         self.load_zbuf = False
    #         self.load_mask = False
    #
    #         self.count_normal = False
    #
    #         self.quantiGTPattern = 'Dataset/model_id:#/id:&_/'
    #
    #         self.compress_ratio_h = 1
    #         self.compress_ratio_w = 1
    #         self.compress_ratio_total = 3
    #
    #         self.img_h = 1448
    #         self.img_w = 2172
    #
    #         ##############################
    #         self.image_size = torch.FloatTensor([[[self.img_w, self.img_h]]])  # the size of the input image
    #         # self.image_size_single = torch.FloatTensor([[[int(self.img_w / self.compress_ratio_total), int(self.img_h / self.compress_ratio_total)]]])  # the size of the input image
    #         self.image_size_single = torch.FloatTensor([[[int(
    #             self.img_w / (self.compress_ratio_w * self.compress_ratio_total)), int(
    #             self.img_h / (self.compress_ratio_h * self.compress_ratio_total))]]])  # the size of the input image
    #
    #         self.use_random_partition = False
    #         self.partition_list = [patch for patch in range(int(self.compress_ratio_w * self.compress_ratio_h))]
    #         self.patch_random_num = 3  # the total num of patches for training, (< compress_ratio ** 2).
    #         #############################
    #
    #         self.render_image_size = 500  # the rendered output size
    #         self.random_render_image_size = [480, 500]
    #         self.blur_radius = 2e-6
    #         self.faces_per_pixel = 3
    #         self.zbuf_front_threshold = 6e-3
    #         self.mesh_radius = self.resol_gt_render * 1.25
    #         # self.mesh_radius4 = self.resoltion_list
    #         self.implicit_texture_D = 0.025
    #         self.D_length = self.resol_gt_render * 3
    #         self.clamp_length = 3.0
    #         self.clamp_length_residual = 5.0
    #         self.D_min_length = 2.0
    #         self.noise_coords_length = 1.0
    #
    #         self.render_image_size_normalize_ratio = self.render_image_size
    #         self.relative_resolution_network_random_ratio = [0.8 ** 2, 6 ** 2]
    #
    #         self.alpha_density_ratio = 10
    #         self.alpha_density_threshold = 0.1
    #         self.siren_omega = 30
    #
    #         self.preprocess_ahead_faces_num = 2
    #         self.preprocess_face_hit_min_num = 10  # 3 / 10
    #         self.load_preprocess_index = False
    #
    #         self.gamma_blend = 1e-4
    #         self.sigma_blend = 1e-4
    #
    #         self.light_condition = '3'
    #         self.light_condition_list = ['3']  # ['0','1','2','3','4','5','6']
    #         self.light_condition_random_list = ['3']  # ['1','2','3','4','5']
    #
    #         self.modelList = ['like']
    #
    #         self.testViewList = [38]  # [38]
    #
    #         # self.trainViewList = range(0, 102, 1)
    #
    #         self.trainViewList = [10, 11, 12, 13, 14, 15, 19, 23, 27, 31, 36, 37, 38, 39, 40, 42, 44, 46, 48, 51]
    #
    #         self.interpolate_novel_view_num = 6
    #         self.interpolate_direction = -1  # 1,-1
    #
    #         self.outer_bound_factor = 0.0
    #
    #         self._cube_D_ = 1
    #         self.center_crop_ratio = 12.0 / 16.0
    #         self.cube_overlapping_ratio_for_training_sample = 0.5
    #
    #         self.estimate_normal = False
    #
    #         self.resol = 0.8
    #         self.random_resol_min = self.resol
    #         self.random_resol_max = self.resol
    #
    #         # self.view_group_center = [0, 2, 5, 7, 10, 13, 16, 19,22, 25,28,31,34,37, 41]  #the center view number of each group pair
    #         # self.view_group_center = [0, 2, 5, 7, 10]  # the center view number of each group pair
    #         self.view_group_center = range(1, 19, 2)
    #         # self.view_group_center = [4, 13, 22, 31, 40, 47]  # the center view number of each group pair
    #         # self.group_pair_index = [[0,2],[1,3],[2,4],[3,5],[4,6],[5,7]]  #the index of the selected group
    #         # self.group_pair_index = [[0, 2], [1, 3], [3, 5], [4, 6], [0, 3], [4, 7],
    #         #                         [2, 5], ]  # the index of the selected group
    #         self.in_group_angle = 360 * 3.1415926535 / 180  # the angle of pairs in a group
    #         self.group_pair_num_max = 4  # the maximum view number of each pair(usually it is 5)
    #         self.group_pair_num_min = 4  # the minimum view number of each pair(usually it is 2)
    #         self.min_group_num = 6  # the minimum group number
    #         self.max_group_num = 6  # the max group number                view_list =
    #
    #         self.augment_group_num = 0
    #         self.sample_alpha = 0.5
    #         self.sample_beta = 0.5
    #         self.sample_delta_alpha = 0.9
    #         self.sample_delta_beta = 0.9
    #
    #         self.delta_length = 20.0
    #
    #         self.rand_angle = 30
    #         self.rand_length = 2 / 1600.0  # 2/1600.0
    #         self.rand_K = 2  # 2
    #
    #         self.z_axis_transform_rate = 1e-2
    #         self.zbuf_min = 40
    #         self.zbuf_max = 80
    #
    #         self.BB_shift = 6.0
    #

    def change_render_mode(self):

        # if self.mode is 'reconstruct':
        #     self.trainViewList = [25]
        #     self.testViewList = [25]  # range(30,36)
        #     self.use_points_blast = False
        #     self.total_gpu_memory = 2e7

        if self.mode is 'refine_points':
            self.use_random_ralative_resolution_ratio = False
            self.testViewList = range(1, 49, 2)  # [13,14,15,16]#[13,14,15,16,24,25,26,27,31,32,33,34]

            self.load_preprocess_index = False

            # self.modelList = [1,4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118]
            self.modelList = [9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]

            self.group_pair_num_max = 1
            self.group_pair_num_min = 1

            self.preprocess_ahead_faces_num = 3
            self.preprocess_face_hit_min_num = 1  # 3 / 10