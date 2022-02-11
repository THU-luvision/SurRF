import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import numpy as np
import pytorch3d
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    SfMPerspectiveCameras,
    SfMOrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    PointsRasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizer,
    #TexturedSoftPhongShader,
    SoftPhongShader,
    SoftGouraudShader
)

from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.points.rasterizer import PointFragments
from pytorch3d.renderer.mesh.utils import _clip_barycentric_coordinates, _interpolate_zbuf
from pytorch3d.renderer.mesh.shading import flat_shading, gouraud_shading, _apply_lighting
#from pytorch3d.renderer.mesh.texturing import interpolate_face_attributes
from pytorch3d.ops import interpolate_face_attributes

from pytorch3d.renderer.blending import BlendParams
# add path for demo utils functions
import sys
import os

import time
import scipy.misc
import matplotlib.pyplot as plt
import pdb
from math import *
from ..models.network  import  RenderNet_C, RenderNet_L, RenderNet_G_old
# sys.path.append(".../")
# from utils.scene import save2ply


class FragmentMachine():
    def __init__(self, params):
        super(FragmentMachine, self).__init__()

        self.params = params

    def random_select(self, fragment):
        if (self.params.rasterizer_mode is 'mesh'):
            # fragment.pix_to_face : shape(N_img, H, W, N_faces)
            # fragment.bary_coords : shape(N_img, H, W, N_faces,3)
            # fragment.zbuf : shape(N_img, H, W, N_faces)
            # fragment.dists : shape(N_img, H, W, N_faces)

            mask = fragment.pix_to_face >= 0 #shape(N_img, H, W, N_faces) # filter out padded faces.
            mask_sum = mask.sum(dim = 3) #shape(N_img, H, W)    # total num of faces hit by each pixel -> N_faces_per_pixel.
            rand_i = (torch.rand(mask_sum.shape).to(mask_sum.device) * mask_sum).type(fragment.pix_to_face.dtype)[...,None] #shape(N_img, H, W)
            # select one of the hit faces for each pixel.
            self.pix_to_face = torch.gather(fragment.pix_to_face, 3, rand_i)
            self.zbuf = torch.gather(fragment.zbuf, 3, rand_i)
            self.dists = torch.gather(fragment.dists, 3, rand_i)
            self.bary_coords = torch.gather(fragment.bary_coords, 3, rand_i[...,None].expand(-1,-1,-1,-1,3))

        elif (self.params.rasterizer_mode is 'pointcloud'):
            # fragment.idx : shape(N_img, H, W, points_per_pixel)
            # fragment.zbuf : shape(N_img, H, W, points_per_pixel)
            # fragment.dists : shape(N_img, H, W, points_per_pixel) # squared Euclidean distance (in NDC units)
            # fragment.perpendicular_coords: shape(N_img, H, W, points_per_pixel, 3)

            mask = fragment.idx >= 0
            mask_sum = mask.sum(dim=3) # shape: (N_img, H, W)
            rand_i = (torch.rand(mask_sum.shape).to(mask_sum.device) * mask_sum).type(fragment.idx.dtype)[...,None] #shape(N_img, H, W)
            self.idx = torch.gather(fragment.idx, 3, rand_i)
            self.zbuf = torch.gather(fragment.zbuf, 3, rand_i)
            self.dists = torch.gather(fragment.dists, 3, rand_i)

    def select_front(self, fragment, view_direction, matrix):
        if (self.params.rasterizer_mode is 'mesh'):
            # fragment.pix_to_face : shape(N_img, H, W, N_faces)
            # fragment.bary_coords : shape(N_img, H, W, N_faces,3)
            # fragment.zbuf : shape(N_img, H, W, N_faces)
            # fragment.dists : shape(N_img, H, W, N_faces)

            mask = fragment.pix_to_face >= 0 #shape(N_img, H, W, N_faces)
            #mask_sum = mask.sum(dim = 3) #shape(N_img, H, W)
            #pdb.set_trace()
            d_res = self.params.zbuf_front_threshold * view_direction.detach()
            d_res = torch.cat((d_res,torch.zeros(d_res.shape[0], 1, d_res.shape[2], d_res.shape[3],d_res.shape[4]).type(d_res.dtype).to(d_res.device)),dim=1)  # shape:(N,4,H,W,N_faces_per_pixel)
            front_threshold = torch.matmul(matrix[:, None, None, None, 2:3, :],d_res.permute(0, 2, 3, 4, 1)[..., None])[:, :, :, :, :,0]  # shape:(N,H,W,N_faces_per_pixel,1)

            # filter the remaining faces whose distance from the first face larger than a threshold.
            # only keep the nearest possible faces (indicate by front_threshold).
            zbuf_mask = (fragment.zbuf[..., 1:] - fragment.zbuf[..., 0:1]) < (front_threshold[:,:,:,1:,0]/self.params.z_axis_transform_rate) # shape:(N_img,H,W,N_faces_per_pixel-1)
            #zbuf_mask = (fragment.zbuf[..., 1:] - fragment.zbuf[..., 0:1]) < self.params.zbuf_front_threshold

            #pdb.set_trace()
            # only keep the nearest faces hit by the pixel.
            mask_zbuf_big = torch.logical_xor(mask[...,1:], zbuf_mask) # shape:(N_img,H,W,N_faces-1)
            self.pix_to_face = fragment.pix_to_face
            self.pix_to_face[...,1:][...,mask_zbuf_big] = -1
            self.zbuf = fragment.zbuf
            self.zbuf[..., 1:][..., mask_zbuf_big] = -1
            self.dists = fragment.dists
            self.dists[..., 1:][..., mask_zbuf_big] = -1
            self.bary_coords = fragment.bary_coords
            self.bary_coords[..., 1:,:][..., mask_zbuf_big,:] = -1

        elif (self.params.rasterizer_mode is 'pointcloud'):
            # fragment.idx : shape(N_img, H, W, points_per_pixel)
            # fragment.zbuf : shape(N_img, H, W, points_per_pixel)
            # fragment.dists : shape(N_img, H, W, points_per_pixel) # squared Euclidean distance (in NDC units)
            # fragment.perpendicular_coords: shape(N_img, H, W, points_per_pixel, 3)

            mask = fragment.idx >= 0  # shape(N_img, H, W, points_per_pixel)
            # pdb.set_trace()
            d_res = self.params.zbuf_front_threshold * view_direction.detach()
            d_res = torch.cat((d_res,
                               torch.zeros(d_res.shape[0], 1, d_res.shape[2], d_res.shape[3], d_res.shape[4]).type(
                                   d_res.dtype).to(d_res.device)), dim=1)  # shape:(N,4,H,W,points_per_pixel)
            front_threshold = torch.matmul(matrix[:, None, None, None, 2:3, :],
                                           d_res.permute(0, 2, 3, 4, 1)[..., None])[:, :, :, :, :,
                              0]  # shape:(N,H,W,points_per_pixel,1)

            # filter the remaining faces whose distance from the first face larger than a threshold.
            # only keep the nearest possible faces (indicate by front_threshold).
            zbuf_mask = (fragment.zbuf[..., 1:] - fragment.zbuf[..., 0:1]) < (front_threshold[:, :, :, 1:,
                                                                              0] / self.params.z_axis_transform_rate)  # shape:(N_img,H,W,N_faces_per_pixel-1)
            # zbuf_mask = (fragment.zbuf[..., 1:] - fragment.zbuf[..., 0:1]) < self.params.zbuf_front_threshold

            # pdb.set_trace()
            # only keep the nearest faces hit by the pixel.
            mask_zbuf_big = torch.logical_xor(mask[..., 1:], zbuf_mask)  # shape:(N_img,H,W,N_points-1)
            self.idx = fragment.idx
            self.idx[..., 1:][..., mask_zbuf_big] = -1
            self.zbuf = fragment.zbuf
            self.zbuf[..., 1:][..., mask_zbuf_big] = -1
            self.dists = fragment.dists
            self.dists[..., 1:][..., mask_zbuf_big] = -1

class RasterizerMachine(nn.Module):
    def __init__(self, params):
        super(RasterizerMachine, self).__init__()

        self.params = params
        # self.meshes = meshes
        self.device = self.params.device
        self.R = torch.FloatTensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])[None, ...]
        self.T = torch.FloatTensor([[0, 0, 0]])
        self.focal_length = torch.FloatTensor([[1, 1]])
        self.principal_point = torch.FloatTensor([[0.0, 0.0]])

        self.fragment = FragmentMachine(self.params)

        self.cameras = SfMOrthographicCameras(device=self.device,
                                              focal_length=self.focal_length,
                                              principal_point=self.principal_point,
                                              R=self.R,
                                              T=self.T)     # defined in NDC space.

        if (self.params.rasterizer_mode is 'mesh'):
            self.raster_settings = RasterizationSettings(
                image_size=self.params.render_image_size,
                blur_radius=self.params.blur_radius,
                faces_per_pixel=self.params.faces_per_pixel,
            )

            self.rasterizer = MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            )
        elif (self.params.rasterizer_mode is 'pointcloud'):
            self.raster_settings = PointsRasterizationSettings(
                image_size=self.params.render_image_size,
                radius=self.params.radius_point,
                points_per_pixel=self.params.points_per_pixel,
            )
            self.rasterizer = PointsRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            )

    def forward(self, representation, matrix, img_size, rasterize_type='test', camera_intrisic=None, camera_position=None):

        if (self.params.rasterizer_mode is 'mesh'):
            mesh = representation
            verts = mesh.verts_packed()  # (V, 3) --> (x, y, z) coordinates of each vertex
            faces = mesh.faces_packed()  # (F, 3) --> indices of the 3 vertices in 'verts' which form the triangular face.
            self.faces_verts = verts[faces]  # (F,3,3)
            self.faces_normals = mesh.faces_normals_packed()

            self.face_num = mesh.faces_padded().shape[1]
            # pdb.set_trace()
            mesh = self.transform_points_mvs(representation=mesh,
                                             matrix=matrix,
                                             img_size=img_size)
            self.matrix = matrix  # camera projection matrix P0.

            self.update_rasterizer(rasterize_type)

            self.camera_intrisic = camera_intrisic * self.render_image_size / self.params.render_image_size_normalize_ratio

            fragments = self.rasterizer(mesh)

            self.bary_coords = fragments.bary_coords
            if self.params.blur_radius > 0.0:
                # TODO: potentially move barycentric clipping to the rasterizer
                # if no downstream functions requires unclipped values.
                # This will avoid unnecssary re-interpolation of the z buffer.

                # self.dists = fragments.dists
                clipped_bary_coords = _clip_barycentric_coordinates(fragments.bary_coords)

                # clipped_bary_coords = fragments.bary_coords
                clipped_zbuf = _interpolate_zbuf(
                    fragments.pix_to_face, clipped_bary_coords, mesh
                )
                fragments = Fragments(
                    bary_coords=clipped_bary_coords,
                    # bary_coords=bary_coords,
                    zbuf=clipped_zbuf,
                    dists=fragments.dists,
                    pix_to_face=fragments.pix_to_face,
                )

            # calculate hit-point-coords (global, local) using barycentric coordinates.
            ## Local.
            pixel_coords_local = interpolate_face_attributes(
                fragments.pix_to_face, fragments.bary_coords - 1.0 / 3.0, self.faces_verts
            )  # shape:(N,H,W,N_faces_per_pixel,3)  # face attribute: (x,y,z) verts coordinates.
            ## Global, world coordinates ?
            pixel_coords = interpolate_face_attributes(
                fragments.pix_to_face, fragments.bary_coords, self.faces_verts
            )  # shape:(N,H,W,N_faces_per_pixel,3)

            ## Global. world coordinates.
            view_direction = pixel_coords - camera_position[:, None, None, None, :]  # shape:(N,H,W,N_faces_per_pixel,3)
            view_direction = F.normalize(view_direction, p=2, dim=-1, eps=1e-8).permute(0, 4, 1, 2,
                                                                                        3).contiguous()  # shape:(N,3,H,W,N_faces_per_pixel)

            if (self.params.faces_per_pixel > 1):
                self.fragment.select_front(fragments, view_direction, self.matrix)
                fragments = self.fragment

            coords = pixel_coords_local.detach().permute(0, 4, 1, 2, 3).contiguous()

            view_num = mesh.faces_padded().shape[0]

            if (view_num == 1):
                pix_to_face_single = (1 + fragments.pix_to_face) % self.face_num
            else:
                pix_to_face_single = (1 + fragments.pix_to_face) % self.face_num

            if (pix_to_face_single.max().item() > mesh.faces_padded().shape[1]):
                raise ValueError('pix_to_face out of range')

            if (self.params.use_relative_resolution):
                reference_view_direction = mesh.faces_normals_packed()[fragments.pix_to_face.detach()].permute(0, 4, 1,
                                                                                                               2,
                                                                                                               3).contiguous()  # shape:(N,3,H,W,N_faces_per_pixel)
                # angle between view_direction and surfel normal.
                cos_alpha = abs(reference_view_direction * view_direction).sum(dim=1)[:, None,
                            ...]  # shape:(N,1,H,W,N_faces_per_pixel)
                relative_resolution = (self.params.mesh_radius * self.camera_intrisic / (fragments.zbuf[:, None,
                                                                                         ...].detach() * self.params.z_axis_transform_rate)) ** 2 * cos_alpha / self.params.relative_resolution_normalize_ratio
            else:
                relative_resolution = None
            if (self.params.use_random_ralative_resolution_ratio):
                random_ratio = random.uniform(self.params.relative_resolution_network_random_ratio[0],
                                              self.params.relative_resolution_network_random_ratio[1])
                relative_resolution *= random_ratio

            fragments.coords = coords  # local. (N,3,H,W,N_faces_per_pixel)
            fragments.view_direction = view_direction
            fragments.pixel_coords_local = pixel_coords_local  # (N,H,W,N_faces_per_pixel,3)
            fragments.pixel_coords = pixel_coords  # global. (N,H,W,N_faces_per_pixel,3)
            fragments.pix_to_face_single = pix_to_face_single
            fragments.relative_resolution = relative_resolution
            fragments.matrix = self.matrix

        elif (self.params.rasterizer_mode is 'pointcloud'):
            point_cloud = representation
            self.points = point_cloud.points_packed()    # (P*N_views, 3)
            self.points_debug = point_cloud.points_padded()   # (N_views, P, 3)
            self.normals = point_cloud.normals_packed()  # (P*N_views, 3)
            # self.normals = point_cloud.normals_padded()

            self.points_num = point_cloud.points_padded().shape[1]

            point_cloud = self.transform_points_mvs(representation=point_cloud,
                                                    matrix=matrix,
                                                    img_size=img_size)

            self.matrix = matrix  # camera projection matrix P0.

            self.update_rasterizer(rasterize_type)

            self.camera_intrisic = camera_intrisic * self.render_image_size / self.params.render_image_size_normalize_ratio

            fragments = self.rasterizer(point_cloud)
            # fragment.idx : shape(N_img, H, W, points_per_pixel)
            # fragment.zbuf : shape(N_img, H, W, points_per_pixel)
            # fragment.dists : shape(N_img, H, W, points_per_pixel) # squared Euclidean distance (in NDC units)
            # fragment.perpendicular_coords: shape(N_img, H, W, points_per_pixel, 3)

            img_coords_global = self.inverse_transform_points_mvs(fragment=fragments,
                                                                  matrix=matrix,
                                                                  img_size=img_size,
                                                                  # z_depth_projection=z_depth_projection
                                                                  ) # shape: (N_views, H, W, N_points, 3)

            mask = (fragments.idx.type(torch.int64) >= 0)[..., None].detach()
            view_direction = img_coords_global - camera_position[:, None, None, None, :]
            # view_direction = view_direction * mask

            point_direction = self.points[fragments.idx.type(torch.int64).detach()] - camera_position[:, None, None, None, :] # shape:(N_views, H, W, N_points, 3)
            point_direction = point_direction * mask

            view_direction = F.normalize(view_direction, p=2, dim=-1, eps=1e-8)

            projection_length = (point_direction * view_direction).sum(dim=-1)

            pixel_coords = camera_position[:, None, None, None, :] + (projection_length[..., None] * view_direction) # shape:(N_views,H,W,N_points,3)
            pixel_coords = pixel_coords * mask

            pixel_coords_local = (pixel_coords - self.points[fragments.idx.type(torch.int64).detach()]) * mask

            view_direction = view_direction.permute(0, 4, 1, 2, 3).contiguous()

            if (self.params.points_per_pixel > 1):
                self.fragment.select_front(fragments, view_direction, self.matrix)
                fragments = self.fragment

            coords = pixel_coords_local.detach().permute(0, 4, 1, 2, 3).contiguous()    # shape(N_views,3,H,W,N_points)

            view_num = point_cloud.points_padded().shape[0]

            if (view_num == 1):
                idx_single = (1 + fragments.idx) % self.points_num
            else:
                idx_single = (1 + fragments.idx) % self.points_num

            if (idx_single.max().item() > point_cloud.points_padded().shape[1]):
                raise ValueError('point idx out of range')

            if (self.params.use_relative_resolution):
                reference_view_direction = point_cloud.normals_packed()[fragments.idx.type(torch.int64).detach(), :].permute(0,4,1,2,3).contiguous()    # shape:(N,3,H,W,N_points)

                # angle between view_direction and surfel normal.
                cos_alpha = abs(reference_view_direction * view_direction).sum(dim=1)[:, None, ...] # shape:(N_views,1,H,W,N_points)

                relative_resolution = (self.params.radius_point * self.camera_intrisic / (fragments.zbuf[:, None, ...].detach() * self.params.z_axis_transform_rate)) ** 2 * cos_alpha / self.params.relative_resolution_normalize_ratio

            else:
                relative_resolution = None

            if (self.params.use_random_ralative_resolution_ratio):
                random_ratio = random.uniform(self.params.relative_resolution_network_random_ratio[0],
                                              self.params.relative_resolution_network_random_ratio[1])
                relative_resolution *= random_ratio

            fragments.coords = coords  # local. (N,3,H,W,N_points)
            fragments.view_direction = view_direction
            fragments.pixel_coords_local = pixel_coords_local  # (N,H,W,N_points,3)
            fragments.pixel_coords = pixel_coords  # global. (N,H,W,N_points,3)
            fragments.pix_to_face_single = idx_single.type(torch.int64)
            # fragments.pix_to_face_single = pix_to_face_single
            fragments.relative_resolution = relative_resolution
            fragments.matrix = self.matrix

        return fragments

    def update_rasterizer(self, rasterize_type='test'):

        if (rasterize_type == 'test'):
            render_image_size = self.params.render_image_size
        elif (rasterize_type == 'train'):
            render_image_size = random.randint(self.params.random_render_image_size[0],
                                               self.params.random_render_image_size[1])

        self.render_image_size = render_image_size

        if (self.params.rasterizer_mode is 'mesh'):
            self.raster_settings = RasterizationSettings(
                image_size=render_image_size,
                blur_radius=self.params.blur_radius,
                faces_per_pixel=self.params.faces_per_pixel,
            )

            self.rasterizer = MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            )
        elif (self.params.rasterizer_mode is 'pointcloud'):
            self.raster_settings = PointsRasterizationSettings(
                image_size=render_image_size,
                radius=self.params.radius_point,
                points_per_pixel=self.params.points_per_pixel,
            )
            self.rasterizer = PointsRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            )

    def transform_points_mvs(self, representation, matrix, img_size):
        '''
        use camera matrix to transform the original global mesh verts
        (keep the dimension of z axis, that means the transformed camera is a SfMOrthographicCameras)
        args:
            mesh: pytorch3d mesh object
            matrix: cameras matrix
                shape:(N_views, 4, 4)
                type: torch.FloatTensor
            img_size: the image size of output image
                shape: (1,1,2)
                type: torch.FloatTensor

        output:
            pytorch3d mesh object
        '''
        if (self.params.rasterizer_mode is 'mesh'):
            mesh = representation
            verts = mesh.verts_padded() # shape: (1, max_num_verts, 3)
            verts_old_packed = mesh.verts_packed() # shape: (V, 3)

            N, P, _ = verts.shape
            ones = torch.ones(N, P, 1, dtype=verts.dtype, device=verts.device)
            verts = torch.cat([verts, ones], dim=2) # shape: (1, max_num_verts, 4)

            verts = torch.matmul(matrix[:, None, :, :], verts[..., None]) # shape: (1, max_num_verts, 4, 1)
            # matrix[:, None, :, :]: (N_views, 1, 4, 4)
            # verts[..., None]: (1, max_num_verts, 4, 1)
            # --> result: (N_views, max_num_verts, 4, 1)

            verts = torch.cat(((2 * verts[:, :, 0:2, 0] / img_size) / verts[:, :, 2:3, 0] - 1,
                               verts[:, :, 2:3, 0] / self.params.z_axis_transform_rate), dim=2) # shape: (N_views, max_num_verts, 3)

            return mesh.offset_verts(verts.view(-1, 3) - verts_old_packed)

        elif (self.params.rasterizer_mode is 'pointcloud'):
            point_cloud = representation
            points = point_cloud.points_padded() # shape: (N_views, P, 3)
            points_old_packed = point_cloud.points_packed() # shape: (N_views * P, 3)
            # points_old_padded = point_cloud.points_padded() # shape: (N_views, P, 3)

            N, P, _ = points.shape

            ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
            points = torch.cat([points, ones], dim=2) # shape: (1, max_num_points, 4)

            points = torch.matmul(matrix[:, None, :, :], points[..., None]) # shape: (N_views, P, 4, 1)

            points = torch.cat(((2 * points[:, :, 0:2, 0] / img_size) / points[..., 2:3, 0] - 1,
                               points[:, :, 2:3, 0] / self.params.z_axis_transform_rate), dim=2)  # shape: (N_views, P, 3)

        return point_cloud.offset(points.view(-1, 3) - points_old_packed)

    def inverse_transform_points_mvs(self, fragment, matrix, img_size, ):
        '''
        use camera matrix to transform the original global mesh verts
        (keep the dimension of z axis, that means the transformed camera is a SfMOrthographicCameras)
        args:
            fragment: pytorch3d fragment object of rasterized points.
            matrix: cameras matrix
                shape:(N_views, 4, 4)
                type: torch.FloatTensor
            img_size: the image size of output image
                shape: (1,1,2)
                type: torch.FloatTensor

        output:
            points: screen points unprojected into 3D global coordinates
                shape: (N, H, W, N_points, 3)
        '''

        points_idx = fragment.idx   # shape: (N, H, W, N_points)
        W, H = img_size[0][0][0].type(torch.int64).item(), img_size[0][0][1].type(torch.int64).item()

        N, H_rast, W_rast, N_points = points_idx.shape

        xx = torch.linspace(0, H, H_rast)
        yy = torch.linspace(0, W, W_rast)

        X, Y = torch.meshgrid(xx, yy)

        X = torch.unsqueeze(X, dim=-1).type(self.points.dtype).to(self.points.device)
        Y = torch.unsqueeze(Y, dim=-1).type(self.points.dtype).to(self.points.device)
        Z = torch.ones(H_rast, W_rast, 1, dtype=self.points.dtype, device=self.points.device)
        tmp = torch.ones(H_rast, W_rast, 1, dtype=self.points.dtype, device=self.points.device)

        img_points = torch.cat((Y, X, Z, tmp), dim=-1)  # shape: (H, W, 4)
        img_points = img_points.expand(N, N_points, H_rast, W_rast, img_points.shape[-1]).permute(0, 2, 3, 1, 4) # (N,H,W,N_points,4)

        inverse_matrix = torch.inverse(matrix)

        points = torch.matmul(inverse_matrix[:, None, None, None, ...], (img_points[..., None])) # shape: (N_views, H, W, N_points, 4, 1)
        # points = torch.matmul(inverse_matrix[:, None, None, None, ...], (img_points * img_points_z_depth)[..., None]) # shape: (N_views, H, W, N_points, 4, 1)
        return points[..., 0:3, 0]  # shape: (N_views, H, W, N_points, 3)

def generate_rotate_vector(v, k, angle):
    angle_pi = angle * np.pi / 180
    rot_x = v[:, 0] * np.cos(angle_pi) + (k[:, 1] * v[:, 2] - k[:, 2] * v[:, 1]) * np.sin(angle_pi)
    rot_y = v[:, 1] * np.cos(angle_pi) + (k[:, 2] * v[:, 0] - k[:, 0] * v[:, 2]) * np.sin(angle_pi)
    rot_z = v[:, 2] * np.cos(angle_pi) + (k[:, 0] * v[:, 1] - k[:, 1] * v[:, 0]) * np.sin(angle_pi)

    rot_v = np.concatenate((rot_x[:, None], rot_y[:, None], rot_z[:, None]), axis=1)
    return rot_v


def generate_tri_mesh(points, normals, radius, colors=None, device=None):
    v = np.concatenate((normals[:, 1][..., None], -normals[:, 0][..., None], np.zeros(normals[:, 0].shape)[..., None]),
                       axis=1)
    v = v / (np.linalg.norm(v, axis=1)[..., None] + 1e-8)
    mesh_points_a = points + radius * v
    mesh_points_b = points + radius * generate_rotate_vector(v, normals, 120)
    mesh_points_c = points + radius * generate_rotate_vector(v, normals, 240)

    mesh_points = np.concatenate((mesh_points_a[:, None, :], mesh_points_b[:, None, :], mesh_points_c[:, None, :]),
                                 axis=1)
    mesh_points = mesh_points.reshape((points.shape[0] * 3, 3))

    face_id = np.array(list(range(points.shape[0] * 3)))
    face_id = face_id.reshape((points.shape[0], 3))

    if (False):
        verts_colour = np.repeat(colors, 3, axis=0)[None, ...]
        verts_texture = pytorch3d.structures.Textures(verts_rgb=torch.FloatTensor(verts_colour).to(device))

        mesh = Meshes(verts=[torch.FloatTensor(mesh_points).to(device)],
                      faces=[torch.FloatTensor(face_id).to(device)],
                      textures=verts_texture)
    else:
        mesh = Meshes(verts=[torch.FloatTensor(mesh_points).to(device)],
                      faces=[torch.FloatTensor(face_id).to(device)],
                      )
    return mesh

def generate_pointcloud(points, normals=None, colors=None, device=None):

    if (False):
        if colors is not None and normals is not None:
            point_cloud = Pointclouds(points=[torch.FloatTensor(points).to(device)],
                                      normals=[torch.FloatTensor(normals).to(device)],
                                      features=[torch.FloatTensor(colors).to(device)])

        elif colors is not None:
            point_cloud = Pointclouds(points=[torch.FloatTensor(points).to(device)],
                                      features=[torch.FloatTensor(colors).to(device)])

        elif normals is not None:
            point_cloud = Pointclouds(points=[torch.FloatTensor(points).to(device)],
                                      normals=[torch.FloatTensor(normals).to(device)])

        else:
            point_cloud = Pointclouds(points=[torch.FloatTensor(points).to(device)])
    else:
        if normals is not None:
            point_cloud = Pointclouds(points=[torch.FloatTensor(points).to(device)],
                                      normals=[torch.FloatTensor(normals).to(device)])
        else:
            point_cloud = Pointclouds(points=[torch.FloatTensor(points).to(device)])

    # print('point_cloud.points_padded().shape: ', point_cloud.points_padded().shape)
    # print('point_cloud.points_padded().shape: ', point_cloud.points_packed().shape)
    return point_cloud
###################################################