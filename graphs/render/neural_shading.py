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
from ..models.network  import  RenderNet_C, RenderNet_L, RenderNet_G_old, ReptileModel
# sys.path.append(".../")
# from utils.scene import save2ply


class ShadingMachine(ReptileModel):
    def __init__(self, params):
        super(ShadingMachine, self).__init__()

        self.params = params
        # self.meshes = meshes
        self.device = self.params.device

        self.render_net_g = RenderNet_G_old(self.params)
        self.render_net_g.to(self.device)
        self.render_net_c = RenderNet_C(self.params)
        self.render_net_c.to(self.device)
        self.render_net_l = RenderNet_L(self.params)
        self.render_net_l.to(self.device)

        self.lighting_condition_id = 0

    def forward(self, fragments, BB, z_embedding,
                lighting_embedding=None, slice_mask=None):

        with torch.no_grad():

            if self.params.mask_render:
                for i in range(fragments.pix_to_face.shape[1]):
                    for j in range(fragments.pix_to_face.shape[2]):
                        for k in range(fragments.pix_to_face.shape[3]):
                            if fragments.pix_to_face[0, i, j, k] != -1 and slice_mask[
                                fragments.pix_to_face[0, i, j, k]] == False:
                                fragments.pix_to_face[0, i, j, k] = -1

        pixel_face_features = z_embedding(fragments.pix_to_face_single).permute(0, 4, 1, 2, 3).contiguous()

        pixel_face_features_d = pixel_face_features[:, 0:self.params.z_length_d, ...]
        pixel_face_features_c = pixel_face_features[:, self.params.z_length_d:(self.params.z_length_d + self.params.z_length_c), ...]

        D = self.render_net_g(pixel_face_features_d, fragments.coords.detach(), fragments.view_direction.detach(), relative_resolution=fragments.relative_resolution)
        self.loss_density = self.render_net_g.loss_density
        self.loss_density2 = self.render_net_g.loss_density2

        d_res = D * fragments.view_direction.detach()

        coords = fragments.pixel_coords_local.detach().permute(0, 4, 1, 2, 3).contiguous() + d_res
        pixel_coords = fragments.pixel_coords.detach().permute(0, 4, 1, 2, 3).contiguous() + d_res  # shape:(N,3,H,W,N_faces_per_pixel)

        d_res = torch.cat((d_res,
                           torch.zeros(d_res.shape[0], 1, d_res.shape[2], d_res.shape[3],
                                       d_res.shape[4]).type(d_res.dtype).to(d_res.device).detach()),
                          dim=1)  # shape:(N,4,H,W,N_faces_per_pixel)

        d_res_out = torch.matmul(fragments.matrix[:, None, None, None, 2:3, :],
                                  d_res.permute(0, 2, 3, 4, 1).contiguous()[..., None])[:, :, :, :,
                     :, 0] / self.params.z_axis_transform_rate  # shape:(N,H,W,N_faces_per_pixel,1)


        if (self.params.rasterizer_mode is 'mesh'):
            self.mask = (fragments.bary_coords.permute(0, 4, 1, 2, 3).sum(dim=1) > 0)[:, None,
                        ...].detach()

        elif (self.params.rasterizer_mode is 'pointcloud'):
            self.mask = (fragments.idx.type(torch.int64) >= 0)[:, None, ...].detach()

        out = self.render_net_c(pixel_face_features_c,
                                coords,
                                n=self.render_net_g.n,
                                relative_resolution=fragments.relative_resolution
                                )

        z, n, ad, s = out
        BB_length = (BB[:, 1] - BB[:, 0] + 1e-6)
        pixel_coords = (self.params.BB_shift + pixel_coords - BB[:, 0][None, :, None, None,
                                                              None]) / BB_length[None, :, None, None, None]
        lighting_feature = lighting_embedding(torch.LongTensor([self.lighting_condition_id]).to(self.device))[..., None, None, None].expand(s.shape[0], -1, s.shape[2], s.shape[3], s.shape[4])
        z = self.render_net_l(z,n,ad,s,
                              self.mask * pixel_coords,
                              # view_direction/F.normalize(BB_length, p=2, dim=0, eps=1e-8)[None, :, None, None, None],
                              fragments.view_direction.detach(),
                              lighting_feature=lighting_feature
                              )


        return z, d_res_out
