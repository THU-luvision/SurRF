import sys


from agents.loss import LossRecorder
from utils.camera import K_partition, partition_image_and_matrix, CameraPOs_as_torch_partitioned, resize_matrix
from dataset.dataset import MVSDataset
from configs.parameter import Params
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




class ReconstructGraph():
    def __init__(self, params):
        super(ReconstructGraph, self).__init__()

        self.params = params

    def create_file(self, model_i):
        self.file_root_reconstruction = os.path.join(self.params.root_file, self.params.renderResultRoot,
                                                str(self.params.modelList[model_i]), 'reconstruction')
        if not os.path.exists(self.file_root_reconstruction):
            os.makedirs(self.file_root_reconstruction)


        self.sample_resolution = self.params.sample_resolution
        self.margin_init = self.params.margin_init
        self.margin_end = self.params.margin_end
        self.ply_filePath = os.path.join(self.file_root_reconstruction,
                                    "%s_sample:%s.ply" % (str(self.params.modelList[model_i]), str(self.sample_resolution)))