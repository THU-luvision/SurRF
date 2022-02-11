import sys
# sys.path.append(".../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.autograd import Variable
import math


class RenderNet_G_old(nn.Module):
    def __init__(self, params):
        super(RenderNet_G_old, self).__init__()
        self.params = params
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        if (self.params.use_relative_resolution):
            self.conv_z_1 = nn.Conv3d(int(self.params.z_length_d) + 2 * 3 + 1, self.params.descriptor_length, 1,
                                      padding=0)
        else:
            self.conv_z_1 = nn.Conv3d(int(self.params.z_length_d) + 2 * 3, self.params.descriptor_length, 1, padding=0)

        self.conv_z_2 = nn.Conv3d(self.params.descriptor_length, self.params.descriptor_length, 1, padding=0)
        self.conv_z_3 = nn.Conv3d(self.params.descriptor_length, self.params.descriptor_length, 1, padding=0)
        self.conv_z_4 = nn.Conv3d(self.params.descriptor_length, 1, 1, padding=0)

        std = math.sqrt(6.0 / self.params.descriptor_length)
        torch.nn.init.normal_(self.conv_z_1.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_z_1.bias, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_z_2.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_z_2.bias, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_z_3.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_z_3.bias, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_z_4.weight, mean=0.0, std=1e-3)
        torch.nn.init.normal_(self.conv_z_4.bias, mean=0.0, std=1e-3)

        if (self.params.use_lighting_mlp):
            self.conv_n = nn.Conv3d(self.params.descriptor_length, 3, 1, padding=0)
            torch.nn.init.normal_(self.conv_n.weight, mean=0.0, std=1e-3)
            torch.nn.init.normal_(self.conv_n.bias, mean=0.0, std=1e-3)

    def forward(self, pixel_face_features, coords, view_direction, return_loss=True, rasterize_type="test",
                relative_resolution=None):
        z = torch.cat((pixel_face_features, coords / self.params.mesh_radius, view_direction), dim=1)

        if (self.params.use_relative_resolution):
            z = torch.cat((z, relative_resolution), dim=1)

        z = torch.sin(self.conv_z_1(z))
        z = torch.sin(self.conv_z_2(z))
        z = torch.sin(self.conv_z_3(z))
        D = self.params.D_length * torch.sin(self.conv_z_4(z))
        if (return_loss):
            self.loss_density = 0.0
            self.loss_density2 = 0.0

        if (self.params.use_lighting_mlp):
            self.n = 0.5 * torch.sin(self.conv_n(z))
        else:
            self.n = None

        return D


class RenderNet_C(nn.Module):
    def __init__(self, params):
        super(RenderNet_C, self).__init__()
        self.params = params

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        # self.softmax = nn.Softmax(dim=2)

        if (self.params.use_feature_alpha):
            self.conv_alpha_1 = nn.Conv3d(int(self.params.descriptor_length_c), 1, 1, padding=0)
            torch.nn.init.normal_(self.conv_alpha_1.weight, mean=0.0, std=1e-3)
            torch.nn.init.normal_(self.conv_alpha_1.bias, mean=0.0, std=1e-3)

        if (self.params.use_relative_resolution):
            self.conv_out_1 = nn.Conv3d(self.params.z_length_c + 1 * 3 + 1, self.params.descriptor_length_c, 1,
                                        padding=0)
        else:
            self.conv_out_1 = nn.Conv3d(self.params.z_length_c + 1 * 3, self.params.descriptor_length_c, 1, padding=0)

        self.conv_out_2 = nn.Conv3d(self.params.descriptor_length_c, self.params.descriptor_length_c, 1, padding=0)
        self.conv_out_3 = nn.Conv3d(self.params.descriptor_length_c, self.params.descriptor_length_c, 1, padding=0)
        self.conv_out_3a = nn.Conv3d(self.params.descriptor_length_c, self.params.descriptor_length_c, 1, padding=0)

        # if (self.params.use_2d_network):
        self.conv_out_4 = nn.Conv3d(self.params.descriptor_length_c, self.params.descriptor_length_c, 1, padding=0)
        self.conv_out_out = nn.Conv3d(self.params.descriptor_length_c, 3, 1, padding=0)
        torch.nn.init.normal_(self.conv_out_out.weight, mean=0.0, std=1e-3)
        torch.nn.init.normal_(self.conv_out_out.bias, mean=0.0, std=1e-3)
        # else:
        #     self.conv_out_4 = nn.Conv3d(self.params.descriptor_length_c, 3, 1, padding=0)

        std = math.sqrt(6.0 / self.params.descriptor_length_c)
        torch.nn.init.normal_(self.conv_out_1.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_out_1.bias, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_out_2.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_out_2.bias, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_out_3.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_out_3.bias, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_out_3a.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_out_3a.bias, mean=0.0, std=std)

        torch.nn.init.normal_(self.conv_out_4.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_out_4.bias, mean=0.0, std=std)

        if (self.params.use_lighting_mlp):
            self.conv_n = nn.Conv3d(self.params.descriptor_length_c, 3, 1, padding=0)
            torch.nn.init.normal_(self.conv_n.weight, mean=0.0, std=1e-1)
            torch.nn.init.normal_(self.conv_n.bias, mean=0.0, std=1e-1)

            self.conv_Cad = nn.Conv3d(self.params.descriptor_length_c, 2 * 3, 1, padding=0)
            torch.nn.init.normal_(self.conv_Cad.weight, mean=0.0, std=1e-3)
            torch.nn.init.normal_(self.conv_Cad.bias, mean=0.0, std=1e-3)
            self.conv_Cs = nn.Conv3d(self.params.descriptor_length_c, 3, 1, padding=0)
            torch.nn.init.normal_(self.conv_Cs.weight, mean=0.0, std=1e-3)
            torch.nn.init.normal_(self.conv_Cs.bias, mean=0.0, std=1e-3)

    def forward(self, pixel_face_features, coords, n=None, relative_resolution=None):

        z = torch.cat((pixel_face_features, coords / self.params.mesh_radius), dim=1)

        if (self.params.use_relative_resolution):
            z = torch.cat((z, relative_resolution), dim=1)

        z = torch.sin(self.conv_out_1(z))
        z = torch.sin(self.conv_out_2(z))
        z = torch.sin(self.conv_out_3(z))
        z = torch.sin(self.conv_out_3a(z))

        if (self.params.use_lighting_mlp):
            # n = 0.5 * torch.sin(self.conv_n(z))
            ad = 0.5 * torch.sin(self.conv_Cad(z))
            s = 0.5 * torch.sin(self.conv_Cs(z))

        if (self.params.use_feature_alpha):
            self.alpha = torch.sigmoid(self.conv_alpha_1(z))
            self.alpha = self.alpha / (self.alpha.max(dim=-1)[0][..., None] + 1e-6)

        z = torch.sin(self.conv_out_4(z))
        z = 0.5 * torch.sin(self.conv_out_out(z))
        # z = 0.5 * torch.sin(self.conv_out_4(z))
        # z = self.conv_out_4(z)

        if (self.params.use_lighting_mlp):
            return (z, n, ad, s)
        else:
            return z


class RenderNet_L(nn.Module):
    def __init__(self, params):
        super(RenderNet_L, self).__init__()
        self.params = params

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(dim=2)

        # self.conv_ad_1 = nn.Conv3d(3 * 3, self.params.descriptor_light_length, 1, padding=0)
        # self.conv_ad_2 = nn.Conv3d(self.params.descriptor_light_length, self.params.descriptor_light_length, 1,padding=0)
        # self.conv_ad_3 = nn.Conv3d(self.params.descriptor_light_length, self.params.descriptor_light_length, 1,padding=0)
        # self.conv_ad_4 = nn.Conv3d(self.params.descriptor_light_length, 3, 1, padding=0)
        #
        std = math.sqrt(6.0 / self.params.descriptor_light_length) / self.params.siren_omega
        # torch.nn.init.normal_(self.conv_ad_1.weight, mean=0.0, std=std)
        # torch.nn.init.normal_(self.conv_ad_1.bias, mean=0.0, std=std)
        # torch.nn.init.normal_(self.conv_ad_2.weight, mean=0.0, std=std)
        # torch.nn.init.normal_(self.conv_ad_2.bias, mean=0.0, std=std)
        # torch.nn.init.normal_(self.conv_ad_3.weight, mean=0.0, std=std)
        # torch.nn.init.normal_(self.conv_ad_3.bias, mean=0.0, std=std)
        # torch.nn.init.normal_(self.conv_ad_4.weight, mean=0.0, std=1e-3)
        # torch.nn.init.normal_(self.conv_ad_4.bias, mean=0.0, std=1e-3)
        if (self.params.use_lighting_embedding):
            self.conv_s_1 = nn.Conv3d(4 * 3 + self.params.embedding_light_length, self.params.descriptor_light_length,
                                      1, padding=0)
        else:
            self.conv_s_1 = nn.Conv3d(4 * 3, self.params.descriptor_light_length, 1, padding=0)
        self.conv_s_2 = nn.Conv3d(self.params.descriptor_light_length, self.params.descriptor_light_length, 1,
                                  padding=0)
        self.conv_s_3 = nn.Conv3d(self.params.descriptor_light_length, self.params.descriptor_light_length, 1,
                                  padding=0)
        self.conv_s_4 = nn.Conv3d(self.params.descriptor_light_length, 3, 1, padding=0)

        torch.nn.init.normal_(self.conv_s_1.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_s_1.bias, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_s_2.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_s_2.bias, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_s_3.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_s_3.bias, mean=0.0, std=std)
        torch.nn.init.normal_(self.conv_s_4.weight, mean=0.0, std=1e-4)
        torch.nn.init.normal_(self.conv_s_4.bias, mean=0.0, std=1e-4)

    def forward(self, z, n, ad, s, pixel_coords, view_direction, lighting_feature=None):
        # ad = torch.cat((ad, n), dim=1)
        # ad = torch.sin(self.conv_ad_1(ad))
        # ad = torch.sin(self.conv_ad_2(ad))
        # ad = torch.sin(self.conv_ad_3(ad))
        # # ad = self.tanh(self.conv_ad_4(ad))
        # ad = 0.5 + 0.5 * torch.sin(self.conv_ad_4(ad))

        s = torch.cat((s, n, pixel_coords, view_direction), dim=1)
        if (self.params.use_lighting_embedding):
            s = torch.cat((s, lighting_feature), dim=1)
        # pdb.set_trace()
        s = torch.sin(self.params.siren_omega * self.conv_s_1(s))
        s = torch.sin(self.params.siren_omega * self.conv_s_2(s))
        # s = torch.sin(self.params.siren_omega * self.conv_s_3(s))
        s = 0.5 * torch.sin(self.params.siren_omega * self.conv_s_4(s))

        # z = ad * (z + 0.5) - 0.5 + s

        if (self.params.lighting_predict_mode == 'total'):
            return z + s
        elif (self.params.lighting_predict_mode == 's'):
            return s
        elif (self.params.lighting_predict_mode == 'z'):
            return z
        elif (self.params.lighting_predict_mode == 'n'):
            return n / 2


class ReptileModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda


class ReptileEmbedding(nn.ModuleList):

    def __init__(self):
        nn.ModuleList.__init__(self)

    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda