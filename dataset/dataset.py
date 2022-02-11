import open3d as o3d
import pdb
import sys
import random
import os

sys.path.append("../")
from utils.scene import initializeCubes, quantizePts2Cubes, quantizeGt_2_ijk, save2ply, quantizeGt_2_ijkxyzrgbn, \
    gen_multi_ijkxyzrgbn, gen_sparse_multi_ijkxyzrgbn
from utils.cams import readAllImages, readAllImagesAllLight
from utils.camera import cameraPs2Ts_all, readCameraP0s_np_allModel, image_compress_coef, resize_matrix, \
    resize_multistage_image_and_matrix, resize_image_and_matrix, generate_multiImageMetaVector, cameraPs2Ts, \
    perspectiveProj, judge_cubic_center_in_view, viewPairAngles_wrt_groupView, select_group, select_group_pairs, \
    generateMetaVector, readCameraRTK_as_np_tanks, K_partition, partition_image_and_matrix, \
    CameraPOs_as_torch_partitioned, resize_matrix, interpolate_cameras
from utils.tools import get_BB_models
from utils.ply import ply2array, o3d_load, zbuf_load, mask_load
from configs.parameter import Params
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
import time


class MVSDataset(Dataset):

    def __init__(self, params, type_data='train'):
        super(MVSDataset, self).__init__()
        self.params = params
        self.type_data = type_data
        self.init_data()

    def init_data(self):

        # print('start init data')
        type_data = self.type_data

        self.BB_models = get_BB_models(
            datasetFolder=self.params.datasetFolder,
            BBNamePattern=self.params.BBNamePattern,
            modelList=self.params.modelList,
            outer_bound_factor=self.params.outer_bound_factor,
            datasetName=self.params._datasetName  # zhiwei
        )

        self.compress_data = {}

        if (type_data is 'train'):
            self.imgs_models_all = readAllImagesAllLight(
                datasetFolder=self.params.datasetFolder,
                datasetName=self.params._datasetName,
                imgNamePattern=self.params.imgNamePattern,
                viewList=self.params.trainViewList,
                modelList=self.params.modelList,
                light_condition_list=self.params.light_condition_list,
            )

            (self.cameraPOs_models, self.cameraPO4s_models, self.cameraRTO4s_models,
             self.cameraKO4s_models) = readCameraP0s_np_allModel(
                datasetFolder=self.params.datasetFolder,
                datasetName=self.params._datasetName,
                poseNamePatternModels=self.params.poseNamePattern,
                modelList=self.params.modelList,
                viewList=self.params.trainViewList)  # (N_views, 3, 4) np

            self.cameraTs_new = cameraPs2Ts_all(self.cameraPOs_models)

            self.cameraPO4s_models_numpy = self.cameraPO4s_models
            self.cameraPO4s_models = torch.from_numpy(self.cameraPO4s_models).type(torch.FloatTensor)
            self.cameraRTO4s_models = torch.from_numpy(self.cameraRTO4s_models).type(torch.FloatTensor)
            self.cameraKO4s_models = torch.from_numpy(self.cameraKO4s_models).type(torch.FloatTensor)
            self.cameraTs_new = torch.from_numpy(self.cameraTs_new).type(torch.FloatTensor)
            self.BB_models_tensor = torch.from_numpy(np.stack(self.BB_models)).type(torch.FloatTensor)

            for index in range(len(self.cameraPOs_models)):
                images = self.imgs_models_all[self.params.light_condition][index]
                BB = self.BB_models[index]
                cameraPOs = self.cameraPOs_models[index]
                # get intrinsic matrix.
                cameraKOs = self.cameraKO4s_models[index]

                self.compress_data[index] = resize_multistage_image_and_matrix(images,
                                                                               cameraPOs,
                                                                               cameraKOs,
                                                                               BB[:, 0],
                                                                               cube_D_mm=self.params._cube_D_ * self.params.resol,
                                                                               _cube_D_=self.params._cube_D_,
                                                                               image_compress_multiple=self.params.image_compress_multiple,
                                                                               image_compress_stage=self.params.image_compress_stage,
                                                                               compress_ratio=self.params.compress_ratio_total
                                                                               )  # return the resized image and camera parameter before go throung 2D network

        elif (type_data is 'test'):
            self.imgs_models_all = readAllImagesAllLight(
                datasetFolder=self.params.datasetFolder,
                datasetName=self.params._datasetName,
                imgNamePattern=self.params.imgNamePattern,
                viewList=self.params.testViewList,
                modelList=self.params.modelList,
                light_condition_list=self.params.light_condition_list,
            )

            (self.cameraPOs_models, self.cameraPO4s_models, self.cameraRTO4s_models,
             self.cameraKO4s_models) = readCameraP0s_np_allModel(
                datasetFolder=self.params.datasetFolder,
                datasetName=self.params._datasetName,
                poseNamePatternModels=self.params.poseNamePattern,
                modelList=self.params.modelList,
                viewList=self.params.testViewList)  # (N_views, 3, 4) np

            self.cameraTs_new = cameraPs2Ts_all(self.cameraPOs_models)

            self.cameraPO4s_models_numpy = self.cameraPO4s_models
            self.cameraPO4s_models = torch.from_numpy(self.cameraPO4s_models).type(torch.FloatTensor)
            self.cameraRTO4s_models = torch.from_numpy(self.cameraRTO4s_models).type(torch.FloatTensor)
            self.cameraKO4s_models = torch.from_numpy(self.cameraKO4s_models).type(torch.FloatTensor)
            self.cameraTs_new = torch.from_numpy(self.cameraTs_new).type(torch.FloatTensor)
            self.BB_models_tensor = torch.from_numpy(np.stack(self.BB_models)).type(torch.FloatTensor)

            if self.params.mode is 'reconstruct':
                self.img_patches_models = []
                self.camP0s_models = []

            elif self.params.mode is 'render_novel_view':
                self.camP0s_models_interpolate = []
                _, cameraPOs_models_interpolate, cameraKOs_models_interpolate, \
                cameraRT4s_models_interpolate, self.cameraTs_interpolate = interpolate_cameras(self.cameraRTO4s_models,
                                                                                          self.cameraKO4s_models,
                                                                                          self.params.inter_choose,
                                                                                          self.params.zoomin_rate,
                                                                                          self.params.interpolate_novel_view_num,
                                                                                          direction=self.params.interpolate_direction,
                                                                                          zoomin_flag=self.params.inter_zoomin)

            for index in range(len(self.cameraPOs_models)):
                images = self.imgs_models_all[self.params.light_condition][index]
                BB = self.BB_models[index]
                cameraPOs = self.cameraPOs_models[index]
                # :
                cameraKOs = self.cameraKO4s_models[index]

                # adjust the function for 'intrinsic matrix'.
                # Adding intrinsic K in both inputs and outputs of 'resize_multistage_image_and_matrix'.
                # this function could be found in 'camera.py'.
                # compress_data[model_i][0][...]: images.
                # compress_data[model_i][1][...]: projection matrix after resizing, (N_views, 3, 4).
                # compress_data[model_i][2][...]: intrinsic matrix K after resizing. (N_views, 3, 3).
                self.compress_data[index] = resize_multistage_image_and_matrix(images,
                                                                               cameraPOs,
                                                                               cameraKOs,
                                                                               BB[:, 0],
                                                                               cube_D_mm=self.params._cube_D_ * self.params.resol,
                                                                               _cube_D_=self.params._cube_D_,
                                                                               image_compress_multiple=self.params.image_compress_multiple,
                                                                               image_compress_stage=self.params.image_compress_stage,
                                                                               compress_ratio=self.params.compress_ratio_total
                                                                               )  # return the resized image and camera parameter before go throung 2D network


                if self.params.mode is 'reconstruct':
                    parti_imgs, _cameraP04s_models = partition_image_and_matrix(
                        torch.from_numpy(self.compress_data[index][0][0]).permute(0, 3, 1, 2),
                        self.compress_data[index][1],
                        self.cameraRTO4s_models[index],
                        self.compress_data[index][2],
                        compress_ratio_h=self.params.compress_ratio_h, compress_ratio_w=self.params.compress_ratio_w)

                    self.img_patches_models.append(parti_imgs)
                    self.camP0s_models.append(_cameraP04s_models)

                elif self.params.mode is 'render_novel_view':


                    cameraP0s_model_interpolate, cameraKOs_model_interpolate = resize_matrix(cameraPOs_models_interpolate[index, :, 0:3, :],
                                                                                             cameraKOs_models_interpolate[index],
                                                                                             self.params.compress_ratio_total)

                    _cameraP04s_model, _, _ = CameraPOs_as_torch_partitioned(cameraP0s_model_interpolate,
                                                                             cameraRT4s_models_interpolate[index],
                                                                             cameraKOs_model_interpolate,
                                                                             compress_ratio_h=self.params.compress_ratio_h,
                                                                             compress_ratio_w=self.params.compress_ratio_w,
                                                                             img_size=((self.params.img_h / self.params.compress_ratio_total),
                                                                                       (self.params.img_w / self.params.compress_ratio_total))
                                                                             )
                    self.camP0s_models_interpolate.append(_cameraP04s_model)

        self.gt_models = o3d_load(datasetName=self.params._datasetName,
                                  datasetFolder=self.params.datasetFolder,
                                  gtNamePattern=self.params.renderGtNamePattern,
                                  modelList=self.params.modelList,
                                  count_normal=self.params.count_normal,
                                  edit_flag=self.params.edit_flag,
                                  )

        if (self.params._datasetName == 'blendedMVS') or (self.params._datasetName == 'tanks_COLMAP'):
            if (self.params.load_view_selection_file):
                self.view_map = []
                for model in self.params.modelList:
                    pair_path = os.path.join(self.params.datasetFolder,
                                             self.params.pairPattern.replace('$', str(model)))
                    total_view_map = self.readPair(pair_path)
                    self.view_map.append(total_view_map[self.params.trainViewList])

    def __len__(self):
        return len(self.params.modelList)

    def __getitem__(self, item):
        sample = self.numpy2tensor(item)
        return sample

    def change_params(self, params):
        self.params = params

    def readPair(self, pair_path):
        with open(pair_path) as f:
            lines = f.readlines()

        view_num = int(lines[0])
        view_map = []
        for i in range(1, view_num + 1):
            cur_line = lines[2 * i].rstrip().split(' ')
            view_pair = [i - 1]
            for j in range(1, int(cur_line[0]) + 1):
                view_pair.append(int(cur_line[2 * j - 1]))
            view_map.append(view_pair)
        view_map = np.array(view_map)

        return view_map

    def get_test_id(self, index, imgs_models_all, cameraPOs_models):

        # id = self.params.modelList[index]
        images = imgs_models_all[self.params.light_condition][index]
        cameraPOs = cameraPOs_models[index]
        BB = self.BB_models[index]
        # gt, gt_rgb = self.gt_models[index]

        # get more outputs from compress_data.
        images_resized, cameraPOs_new, cameraKOs_new, (compress_h_new, compress_w_new) = self.compress_data[index]

        cameraPOs_network, _ = resize_matrix(cameraPOs_new, cameraKOs_new,
                                             self.params.image_compress_multiple)  # return the camera parameter after go throung 2D network


        cameraTs_new = cameraPs2Ts(cameraPOs_new)  # (N_view, 3)

        return (images_resized)

    def get_id(self, index, imgs_models_all, cameraPOs_models):

        # id = self.params.modelList[index]
        images = imgs_models_all[self.params.light_condition][index]
        cameraPOs = cameraPOs_models[index]
        BB = self.BB_models[index]
        if (self.params._datasetName == 'blendedMVS') or (self.params._datasetName == 'tanks_COLMAP'):
            if (self.params.load_view_selection_file):
                view_map = self.view_map[index]
        # gt, gt_rgb = self.gt_models[index]

        # get more outputs from compress_data.
        images_resized, cameraPOs_new, cameraKOs_new, (compress_h_new, compress_w_new) = self.compress_data[index]

        cameraTs_new = cameraPs2Ts(cameraPOs_new)

        while True:

            t_start = time.time()

            if (self.params.load_view_selection_file):

                num_list = list(range(view_map.shape[0]))
                select_num = random.sample(num_list, self.params.min_group_num)
                temp_view_map = self.view_map[0][select_num]

                view_list = []
                for i in range(temp_view_map.shape[0]):
                    # print(temp_view_map)
                    num_list2 = list(range(1, temp_view_map.shape[1]))
                    select_num2 = random.sample(num_list2, self.params.group_pair_num_max)
                    view_list.append(temp_view_map[i, select_num2])

            else:
                view_list = select_group(projection_M=cameraPOs_new,
                                         cameraTs=cameraTs_new,
                                         group_cameraTs=cameraTs_new[self.params.view_group_center, :],
                                         # xyz_3D=cubic_param_np['xyz'] + 0.5 * (1-self.params.center_crop_ratio) * self.params._cube_D_ * self.params.resol,
                                         xyz_3D=BB.mean(axis=1),
                                         cube_length=0.01,
                                         image_shape=images_resized[0].shape[1:3],
                                         angle_thres=self.params.in_group_angle,
                                         group_pair_num_max=self.params.group_pair_num_max,
                                         group_pair_num_min=self.params.group_pair_num_min,
                                         )

            t_end = time.time()

            if (len(view_list) < self.params.min_group_num):
                continue
            else:

                if (len(view_list) > self.params.max_group_num):
                    view_list = random.sample(view_list, k=self.params.max_group_num)

                return (images_resized, view_list)

    def numpy2tensor(self, index):

        sample = {}
        sample['train'] = {}
        sample['test'] = {}
        sample['augment'] = {}

        sample['train']['image'] = {}
        sample['train']['meta'] = {}
        sample['train']['zbuf'] = {}
        sample['train']['mask'] = {}
        ############################
        sample['train']['img_patches'] = {}
        sample['train']['camP0s_list'] = {}
        ############################

        sample['test']['image'] = {}
        sample['test']['meta'] = {}
        sample['test']['zbuf'] = {}
        sample['test']['mask'] = {}
        ############################
        sample['test']['img_patches'] = {}
        sample['test']['camP0s_list'] = {}
        ############################

        sample['augment']['image'] = {}
        sample['augment']['meta'] = {}
        sample['augment']['zbuf'] = {}
        sample['augment']['mask'] = {}
        ############################
        sample['augment']['img_patches'] = {}
        sample['augment']['camP0s_list'] = {}
        ############################

        if (self.type_data is 'train'):

            (images_resized, view_list) = self.get_id(index, self.imgs_models_all, self.cameraPOs_models)

            for i, group_id in enumerate(view_list):
                sample['train']['image'][i] = {}
                sample['train']['meta'][i] = {}
                sample['train']['zbuf'][i] = {}
                sample['train']['mask'][i] = {}
                ##################################
                sample['train']['img_patches'][i] = {}
                sample['train']['camP0s_list'][i] = {}
                sample['train']['meta'][i]['view_list'] = torch.from_numpy(np.array(group_id)).type(torch.LongTensor)
                ##################################

                for stage in range(self.params.image_compress_stage):
                    sample['train']['image'][i][stage] = torch.from_numpy(
                        np.moveaxis(images_resized[stage][group_id, :, :, :], -1, 1)).type(torch.FloatTensor)

                    sample['train']['img_patches'][i], sample['train']['camP0s_list'][i] = partition_image_and_matrix(
                        sample['train']['image'][i][stage],
                        self.compress_data[index][1][
                            np.array(group_id).tolist(), ...],
                        self.cameraRTO4s_models[index][
                            np.array(group_id).tolist(), ...],
                        self.compress_data[index][2][
                            np.array(group_id).tolist(), ...],
                        compress_ratio_h=self.params.compress_ratio_h, compress_ratio_w=self.params.compress_ratio_w)

        elif (self.type_data is 'test'):
            (images_resized_test) = self.get_test_id(index, self.imgs_models_all, self.cameraPOs_models)

            for stage in range(self.params.image_compress_stage):
                sample['test']['image'][stage] = torch.from_numpy(np.moveaxis(images_resized_test[stage], -1, 1)).type(
                    torch.FloatTensor)

                sample['test']['img_patches'], sample['test']['camP0s_list'] = partition_image_and_matrix(
                    torch.from_numpy(sample['test']['image'][stage]).permute(0, 3, 1, 2),
                    self.compress_data[index][1],
                    self.cameraRTO4s_models[index],
                    self.compress_data[index][2],
                    compress_ratio_h=self.params.compress_ratio_h, compress_ratio_w=self.params.compress_ratio_w)

        return sample

    def test(self):
        name = '/home/frank/aMyLab/sgan/inputs/DTU_MVS/preprocess/ply/9/2.0_surface_xyz_flow.ply'
        pcd = o3d.io.read_point_cloud(name)

        points = np.asarray(pcd.points)
        ones = np.ones((points.shape[0], 1))
        points = np.concatenate((points, ones), axis=1)

        # pdb.set_trace()
        z_big_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1000, 0], [0, 0, 0, 1]])
        cameraPO4s_models_numpy_new = np.matmul(z_big_matrix[None, ...], self.cameraPO4s_models_numpy[0, :, :, :])
        points = np.matmul(cameraPO4s_models_numpy_new[:, None, :, :], points[None, :, :, None])
        pdb.set_trace()
        pcd.points = o3d.utility.Vector3dVector(points[0, :, :3, 0])
        # pdb.set_trace()
        vis = o3d.visualization.Visualizer()

        camera_parameter = o3d.camera.PinholeCameraParameters()

        camera_parameter.extrinsic = self.cameraPO4s_models_numpy[0, 0]
        camera_parameter.intrinsic.set_intrinsics(800, 800, 1, 1, 0, 0)
        # pdb.set_trace()

        vis.create_window(width=800, height=800, left=0, top=0, visible=True)
        # vis.create_window()
        vis.add_geometry(pcd)
        vis.run()

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_parameter)
        depth = vis.capture_depth_float_buffer(True)
        vis.destroy_window()
        plt.imshow(np.asarray(depth))
        plt.show()
        pdb.set_trace()


if __name__ == '__main__':
    params = Params()
    dataset = MVSDataset(params, type_data='train')
    # dataset.test()

    sample = dataset.numpy2tensor(0)
