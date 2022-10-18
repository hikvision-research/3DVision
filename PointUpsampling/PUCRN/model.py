from collections import OrderedDict
import os
import torch
from math import log
from collections import defaultdict


from network.model_loss import ChamferLoss
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation



class Model(object):
    def __init__(self, net, phase, opt):
        self.net = net
        self.phase = phase

        if self.phase == 'train':
            self.error_log = defaultdict(int)
            self.chamfer_criteria = ChamferLoss()
            self.old_lr = opt.lr_init
            self.lr = opt.lr_init
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=opt.lr_init,
                                                    betas=(0.9, 0.999))
            self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.7)
            self.decay_step = opt.decay_iter 

        self.step = 0

    def set_input(self, input_pc, radius, label_pc=None):
        """`
        :param
            input_pc       Bx3xN
            up_ratio       int
            label_pc       Bx3xN'
        """
        self.input = input_pc.detach()
        self.radius = radius
        # gt point cloud
        if label_pc is not None:
            self.gt = label_pc.detach()
        else:
            self.gt = None

    def forward(self):
        if self.gt is not None:
            self.predicted, self.gt = self.net(self.input,  gt=self.gt)
        else:
            self.predicted = self.net(self.input)
        
    def get_lr(self, optimizer):
        """Get the current learning rate from optimizer.
        """
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def optimize(self, steps=None, epoch=None):
        """
        run forward and backward, apply gradients
        """
        self.optimizer.zero_grad()
        self.net.train()
        self.forward()

        P1, P2, P3  = self.predicted
        gt_downsample_idx = furthest_point_sample(self.gt.permute(0, 2, 1).contiguous(), P1.shape[1])
        gt_downsample = gather_operation(self.gt, gt_downsample_idx)
        cd_1 =  self.compute_chamfer_loss(P1, gt_downsample) 
        cd_2 =  self.compute_chamfer_loss(P2, self.gt) 
        cd_3 =  self.compute_chamfer_loss(P3, self.gt) 
        loss = cd_1 + cd_2 + cd_3 
        losses = [cd_1, cd_2, cd_3]

        loss.backward()
        self.optimizer.step()
        if steps % self.decay_step == 0 and steps!=0:
            self.lr_scheduler.step()

        lr = self.get_lr(self.optimizer)
        return losses, lr

    def compute_chamfer_loss(self, pc, pc_label):

        loss_chamfer = self.chamfer_criteria(
            pc.transpose(1, 2).contiguous(),
            pc_label.transpose(1, 2).contiguous(), self.radius)

        return loss_chamfer
