import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import numpy as np
import pytorch3d
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    SfMPerspectiveCameras,
    SfMOrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    #TexturedSoftPhongShader,
    SoftPhongShader,
    SoftGouraudShader
)

from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.utils import _clip_barycentric_coordinates, _interpolate_zbuf
from pytorch3d.renderer.mesh.shading import flat_shading, gouraud_shading, _apply_lighting
#from pytorch3d.renderer.mesh.texturing import interpolate_face_attributes
from pytorch3d.ops import interpolate_face_attributes

from pytorch3d.renderer.blending import BlendParams
# add path for demo utils functions
import sys
import os

import time

import pdb
from math import *
from ..models.network  import   RenderNet_C, RenderNet_L, RenderNet_G_old
# sys.path.append(".../")
# from utils.scene import save2ply
from .rasterizer import RasterizerMachine
from .neural_shading import ShadingMachine
from ..models.network  import  ReptileModel

class RenderMachine(ReptileModel):
    def __init__(self, params, device):
        super(RenderMachine, self).__init__()

        self.params = params
        # self.meshes = meshes
        self.device = self.params.device

        self.rasterizer_machine = RasterizerMachine(self.params)
        self.rasterizer_machine.to(device)
        self.shading_machine = ShadingMachine(self.params)
        self.shading_machine.to(device)

    def clone(self):
        clone = RenderMachine(self.params, self.device)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

    def forward(self, mesh, matrix, camera_position, BB, img_size, z_embedding, out_img_size, rasterize_type='test',
                camera_intrisic=None, lighting_embedding=None, slice_mask=None):

        self.rasterize_type = rasterize_type

        with torch.no_grad():

            fragments = self.rasterizer_machine(representation = mesh, matrix = matrix, img_size = img_size,  rasterize_type = rasterize_type, camera_intrisic = camera_intrisic, camera_position = camera_position)

        pixel_features, d_res_out = self.shading_machine(fragments = fragments, BB = BB, z_embedding = z_embedding, lighting_embedding=lighting_embedding, slice_mask=slice_mask)

        features = self.simple_blend(pixel_features, fragments.zbuf.detach() + d_res_out[..., 0], fragments)
        mask, images = features
        if (self.params.render_ob_flag != 0):
            mask = mask * self.mask_list[self.params.render_ob_flag - 1]

        self.zbuf = fragments.zbuf[:, :, :, 0:1].detach()

        images = F.interpolate(images, size=out_img_size, mode='bilinear', align_corners=False)  # (N_img, 3, H, W)
        mask = F.interpolate(0.0 + mask, size=out_img_size)  # (N_img, 1, H, W)
        zbuf = F.interpolate((self.zbuf + d_res_out[:, :, :, 0:1, 0]).permute(0, 3, 1, 2).contiguous(),
                             size=out_img_size, mode='bilinear', align_corners=False)  # (N_img, 1, H, W)


        return (mask.detach(), images, zbuf.detach())

    def simple_blend(self, pixel_features, pixel_zbufs, fragments):

        N, H, W, K = fragments.zbuf.shape
        device = fragments.zbuf.device

        # Mask for padded pixels.
        if (self.params.rasterizer_mode is 'mesh'):
            mask = fragments.pix_to_face >= 0
        elif (self.params.rasterizer_mode is 'pointcloud'):
            mask = fragments.idx >= 0

        num_pixel = mask.sum(dim=-1)
        num_pixel[num_pixel == 0] = -1
        mask_total = (num_pixel > 0)[:, None, ...].detach()  # (N_img, 1, H, W)

        pixel_zbufs_mean = (mask * pixel_zbufs).sum(dim=-1) / num_pixel
        self.pixel_zbufs_var = (mask * ((pixel_zbufs - pixel_zbufs_mean[..., None]) ** 2)).mean()


        pixel_features_mean = (mask[:, None, ...] * pixel_features).sum(dim=-1) / num_pixel[:, None, :, :]
        self.pixel_features_var = (
                    mask[:, None, ...] * ((pixel_features - pixel_features_mean[..., None]) ** 2)).mean()

        return (mask_total, pixel_features_mean)

    def points_generation(self, meshes, z_embedding, sample_resolution=10, margin_init=0.2, margin_end=0.8,
                          camera_position=None, lighting_embedding=None, BB=None):

        with torch.no_grad():
            faces = meshes.faces_padded()[0][:-1, :]  # (N_view, F, 3)
            verts = meshes.verts_padded()[0][:-3, :]  # (N_view, v, 3)

            face_verts = verts[faces, :]  # (F,3,3)
            # pdb.set_trace()

            XYZ = generate_meshgrid(sample_resolution, margin_init, margin_end)  # numpy shape:(N_interpolate,3)
            XYZ = torch.from_numpy(XYZ).type(face_verts.dtype)

            points_coords_global = (face_verts[:, None, :, :] * XYZ[None, :, :, None]).sum(dim=2)  # (F,N_interpolate,3)
            points_coords_local = points_coords_global - face_verts[:, None, :, :].mean(dim=2)  # (F,N_interpolate,3)

            reference_view_direction = points_coords_global - camera_position[None, :]  # (F,N_interpolate,3)
            zbuf = (reference_view_direction ** 2).sum(dim=2)
            # reference_view_direction = meshes.faces_normals_padded()[0][:-1, None, :].expand(-1,points_coords_local.shape[1],-1)  # (F,N_interpolate,3)
            # pdb.set_trace()
            reference_view_direction = F.normalize(reference_view_direction, p=2, dim=-1, eps=1e-8)

            normal_view_direction = meshes.faces_normals_packed()[:-1, :]  # (F, 3)
            normal_view_direction = normal_view_direction[:, None, :].expand(-1, points_coords_local.shape[1],
                                                                             -1)  # (F,N_interpolate,3)
            cos_alpha = abs(normal_view_direction * reference_view_direction).sum(dim=2)[:, None, ...]  # (F,1,3)

            # relative_resolution = (self.params.mesh_radius*2892.33007/(zbuf[:,None,...].detach()* self.params.z_axis_transform_rate))**2 * cos_alpha / self.params.relative_resolution_normalize_ratio #(F,1,N_interpolate)
            relative_resolution0 = torch.FloatTensor(
                [50 * sample_resolution ** 2 / self.params.relative_resolution_normalize_ratio])
            relative_resolution0 = relative_resolution0.expand(cos_alpha.shape)

            #total_index = self.get_index(meshes, relative_resolution0, points_coords_local, face_verts).to(self.params.device)

            # pixel_face_features_d = z_embedding(total_index)[..., 0:self.params.z_length_d]
            # pixel_face_features_c = z_embedding(total_index)[...,self.params.z_length_d:(self.params.z_length_d + self.params.z_length_c)]
            pixel_face_features_d = z_embedding.weight[1:,:][:face_verts.shape[0],0:self.params.z_length_d] #(F,N_feature_d)
            pixel_face_features_c = z_embedding.weight[1:,:][:face_verts.shape[0],self.params.z_length_d:(self.params.z_length_d + self.params.z_length_c)]#(F,N_feature_c)

            pixel_face_features_d = pixel_face_features_d[:,None,:].expand(-1, points_coords_local.shape[1],-1) #(F,N_interpolate,n_feature_d)
            pixel_face_features_c = pixel_face_features_c[:,None,:].expand(-1, points_coords_local.shape[1],-1) #(F,N_interpolate,n_feature_c)
            # pdb.set_trace()

            N_points_single = int(self.params.total_gpu_memory / (points_coords_global.shape[1] * (
                        pixel_face_features_d.shape[1] + pixel_face_features_c.shape[1])))
            N_iter = int(points_coords_global.shape[0] / N_points_single)
            # pdb.set_trace()

            points_coords_global_new = torch.zeros(points_coords_global.shape).reshape(-1, 3)
            points_colour_global_new = torch.zeros(points_coords_global_new.shape)
            points_normal_global_new = torch.zeros(points_coords_global_new.shape)

            for group_i in range(int(points_coords_global.shape[0] // N_points_single) + 1):
                torch.cuda.empty_cache()
                id_init = N_points_single * group_i
                id_end = N_points_single * (group_i + 1) if (
                            group_i is not (int(points_coords_global.shape[0] // N_points_single))) else \
                points_coords_global.shape[0]
                # pdb.set_trace()
                if (id_init == id_end):
                    break
                pixel_face_features_d_local = \
                pixel_face_features_d[id_init:id_end].reshape(-1, pixel_face_features_d.shape[2])[
                    ..., None, None, None].to(self.device)  # (N,n_feature_d,1,1,1)
                pixel_face_features_c_local = \
                pixel_face_features_c[id_init:id_end].reshape(-1, pixel_face_features_c.shape[2])[
                    ..., None, None, None].to(self.device)  # (N,n_feature_c,1,1,1)
                points_coords_local_local = points_coords_local[id_init:id_end].reshape(-1, 3)[
                    ..., None, None, None].to(self.device)  # (N,3,1,1,1)
                points_coords_global_local = points_coords_global[id_init:id_end].reshape(-1, 3)[
                    ..., None, None, None].to(self.device)  # (N,3,1,1,1)
                reference_view_direction_local = reference_view_direction[id_init:id_end].reshape(-1, 3)[
                    ..., None, None, None].to(self.device)  # (N,1,1,1,1)

                relative_resolution = torch.FloatTensor(
                    [sample_resolution ** 2 / self.params.relative_resolution_normalize_ratio])
                # pdb.set_trace()
                relative_resolution = relative_resolution.expand(pixel_face_features_d_local.shape[0], 1,
                                                                 pixel_face_features_d_local.shape[2],
                                                                 pixel_face_features_d_local.shape[3],
                                                                 pixel_face_features_d_local.shape[4]).to(
                    pixel_face_features_d_local.device)

                D = self.shading_machine.render_net_g(pixel_face_features_d_local, points_coords_local_local.detach(),
                                      reference_view_direction_local.detach(), return_loss=False,
                                      relative_resolution=relative_resolution)  # shape:(N,1,H,W,N_faces_per_pixel)

                points_coords_global_local_new = points_coords_global_local + D * reference_view_direction_local

                if (self.params.use_points_blast):
                    points_coords_global_local_new_blast = points_coords_global_local_new - 30 * self.params.mesh_radius * reference_view_direction_local
                    # pdb.set_trace()
                    points_coords_global_new[
                    (id_init * points_coords_local.shape[1]):(id_end * points_coords_local.shape[1]),
                    :] = points_coords_global_local_new_blast[:, :, 0, 0, 0].cpu()
                else:
                    points_coords_global_new[
                    (id_init * points_coords_local.shape[1]):(id_end * points_coords_local.shape[1]),
                    :] = points_coords_global_local_new[:, :, 0, 0, 0].cpu()
                # pdb.set_trace()
                if (self.params.return_points_colour):
                    points_coords_local_local_new = points_coords_local_local + D * reference_view_direction_local
                    out = self.shading_machine.render_net_c(pixel_face_features_c_local, points_coords_local_local_new,
                                            n=self.shading_machine.render_net_g.n, relative_resolution=relative_resolution)

                    if (self.params.use_lighting_mlp):
                        z, n, ad, s = out
                        BB_length = (BB[:, 1] - BB[:, 0] + 1e-6)
                        pixel_coords = (self.params.BB_shift +
                                        points_coords_global_local_new -
                                        BB[:, 0][None, :, None, None, None]) / BB_length[None, :, None, None, None]

                        lighting_feature = \
                        lighting_embedding(torch.LongTensor([self.shading_machine.lighting_condition_id]).to(self.device))[
                            ..., None, None, None].expand(s.shape[0], -1, s.shape[2], s.shape[3], s.shape[4])

                        z = self.shading_machine.render_net_l(z, n, ad, s, pixel_coords,
                                                              reference_view_direction_local,
                                                              lighting_feature=lighting_feature
                                                              )
                        if (self.params.use_2d_network):
                            z = z[0]
                        points_colour_global_new[
                        (id_init * points_coords_local.shape[1]):(id_end * points_coords_local.shape[1]), :] = 255 * (
                                    0.5 + z[:, :, 0, 0, 0]).cpu()
                        points_normal_global_new[
                        (id_init * points_coords_local.shape[1]):(id_end * points_coords_local.shape[1]), :] = n[:, :,
                                                                                                               0, 0,
                                                                                                               0].cpu()
                        index1 = np.where(points_colour_global_new < 0)
                        index2 = np.where(points_colour_global_new > 255)
                        points_colour_global_new[index1] = 0
                        points_colour_global_new[index2] = 255
                    else:
                        z = out
                        if (self.params.use_2d_network):
                            z = z[0]
                        points_colour_global_new[
                        (id_init * points_coords_local.shape[1]):(id_end * points_coords_local.shape[1]), :] = 255 * (
                                0.5 + z[:, :, 0, 0, 0]).cpu()
            if (self.params.return_points_colour):
                return points_coords_global_new, points_colour_global_new, points_normal_global_new
            else:
                return points_coords_global_new



def generate_meshgrid(sample_resolution = 10, margin_init = -0.0, margin_end = 1.0):
    x = np.arange(margin_init, margin_end + 1.0/sample_resolution, 1.0/sample_resolution)
    y = np.arange(margin_init, margin_end + 1.0/sample_resolution, 1.0/sample_resolution)
    #x = np.array([0.5])
    #y = np.array([0.5])
    #z = 1 - x - y
    #pdb.set_trace()

    #xx, yy, zz = np.meshgrid(x, y)
    #XYZ = np.array([yy.flatten(), xx.flatten(), zz.flatten()]).reshape(3, x.shape[0], x.shape[0], x.shape[0])
    #XYZ = np.moveaxis(XYZ, 0, 3) #numpy shape:(s,s,s,3)
    #XYZ = np.array([yy.flatten(), xx.flatten(), zz.flatten()]).reshape(3,-1)

    xx, yy = np.meshgrid(x, y)
    XY = np.array([yy.flatten(), xx.flatten()]).reshape(2, -1)
    XY = XY[:,(XY.sum(axis=0) >= (1-margin_end)) * (XY.sum(axis=0) <= (1-margin_init))]
    XYZ = np.concatenate((XY, 1-XY.sum(axis=0)[None,:]), axis=0) #numpy shape:(3, N_interpolate)

    return np.moveaxis(XYZ, 0, 1)  #numpy shape:(N_interpolate, 3)


