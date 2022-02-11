from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import numpy as np

class LossRecorder(nn.Module):
    def __init__(self, params):
        super(LossRecorder, self).__init__()

        self.params = params

        self.loss_dict = {}
        self.loss_dict['loss_img'] = 0.0
        self.loss_dict['loss_zbuf'] = 0.0

        self.avg_loss_list = []
        for i in range(len(self.loss_dict)):
            self.avg_loss_list.append(0.0)

        self.loss_count = 0

    def add_loss(self, loss_list):

        for i, loss_ele in enumerate(loss_list):
            self.avg_loss_list[i] += loss_ele.item()

        self.loss_count += 1

    def forward(self, loss_list):

        loss_total = self.params.loss_img_weight * loss_list[0] + self.params.loss_zbuf_weight * loss_list[1]
        self.add_loss(loss_list)
        return loss_total

    def update_loss_dict(self, loss_list):
        self.loss_dict['loss_img'] = loss_list[0]
        self.loss_dict['loss_zbuf'] = loss_list[1]


    def return_avg(self, summary_writer: object = None, current_iteration: object = None) -> object:

        self.avg_loss_list_out = []

        if(self.loss_count == 0):
            print('error occur in loss record')
            return 0
        #print('self.avg_loss_list', self.avg_loss_list)
        for i, loss_ele in enumerate(self.avg_loss_list):
            self.avg_loss_list_out.append(self.avg_loss_list[i]/self.loss_count)
            self.avg_loss_list[i] = 0.0

        #print('avg_loss_list_out',avg_loss_list_out)
        #print('self.loss_count', self.loss_count)
        # if(summary_writer is not None):
        #     summary_writer.add_scalar("epoch/loss", self.avg_loss_list_out[0], current_iteration)
        #     summary_writer.add_scalar("epoch/loss zbuf", self.avg_loss_list_out[1], current_iteration)

        self.loss_count = 0
        return self.avg_loss_list_out



    def clear_loss(self):
        pass