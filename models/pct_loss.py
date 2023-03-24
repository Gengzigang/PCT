# --------------------------------------------------------
# Pose Compositional Tokens
# Written by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.models.builder import LOSSES


@LOSSES.register_module()
class JointS1Loss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def smooth_l1_loss(self, pred, gt):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < self.beta
        loss = torch.where(cond, 0.5*l1_loss**2/self.beta, l1_loss-0.5*self.beta)
        return loss

    def forward(self, pred, gt):

        joint_dim = gt.shape[2] - 1
        visible = gt[..., joint_dim:]
        pred, gt = pred[..., :joint_dim], gt[..., :joint_dim]
 
        loss = self.smooth_l1_loss(pred, gt) * visible
        loss = loss.mean(dim=2).mean(dim=1).mean(dim=0)

        return loss


@LOSSES.register_module()
class Tokenizer_loss(nn.Module):
    def __init__(self, joint_loss_w, e_loss_w, beta=0.05):
        super().__init__()

        self.joint_loss = JointS1Loss(beta)
        self.joint_loss_w = joint_loss_w

        self.e_loss_w = e_loss_w

    def forward(self, output_joints, joints, e_latent_loss):

        losses = []
        joint_loss = self.joint_loss(output_joints, joints)
        joint_loss *= self.joint_loss_w
        losses.append(joint_loss)

        e_latent_loss *= self.e_loss_w
        losses.append(e_latent_loss)

        return losses


@LOSSES.register_module()
class Classifer_loss(nn.Module):
    def __init__(self, token_loss=1.0, joint_loss=1.0, beta=0.05):
        super().__init__()

        self.token_loss = nn.CrossEntropyLoss()
        self.token_loss_w = token_loss

        self.joint_loss = JointS1Loss(beta=beta)
        self.joint_loss_w = joint_loss

    def forward(self, p_logits, p_joints, g_logits, joints):

        losses = []
        if self.token_loss_w > 0:
            token_loss = self.token_loss(p_logits, g_logits)
            token_loss *= self.token_loss_w
            losses.append(token_loss)
        else:
            losses.append(None)
        
        if self.joint_loss_w > 0:
            joint_loss = self.joint_loss(p_joints, joints)
            joint_loss *= self.joint_loss_w
            losses.append(joint_loss)
        else:
            losses.append(None)
            
        return losses