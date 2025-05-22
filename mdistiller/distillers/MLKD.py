from termios import CEOL
from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller
from .loss import CrossEntropyLabelSmooth

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, reduce=True, logit_stand=False):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd


def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_conf(x, y, lam, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = lam.reshape(-1,1,1,1)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class MLKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(MLKD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND 
        self.trainer = cfg.SOLVER.TRAINER

        #deepkd
        self.alpha = cfg.SOLVER.DEEPKD.ALPHA
        self.beta = cfg.SOLVER.DEEPKD.BETA
        self.warmup_factor = cfg.SOLVER.DEEPKD.WARMUP_FACTOR
        self.dkd = cfg.SOLVER.DEEPKD.DKD

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)

        batch_size, class_num = logits_student_strong.shape

        pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()

        class_confidence = torch.sum(pred_teacher_weak, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), 50
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()
        if self.trainer == 'deepkd' or self.trainer == 'deepkd_aug':
            loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))
            #weak
            loss_tckd_weak_1,loss_nckd_weak_1 = self.deepkd_plugin(
                logits_student_weak, 
                logits_teacher_weak, 
                target,
                alpha=self.alpha,
                beta=self.beta,
                temperature=self.temperature,
                warmup_factor=self.warmup_factor,
                DKD=self.dkd,
                **kwargs)
            loss_tckd_weak_1 = self.kd_loss_weight * (loss_tckd_weak_1 * mask).mean()
            loss_nckd_weak_1 = self.kd_loss_weight * (loss_nckd_weak_1 * mask).mean()
            loss_tckd_weak_2,loss_nckd_weak_2 = self.deepkd_plugin(
                logits_student_weak, 
                logits_teacher_weak, 
                target,
                alpha=self.alpha,
                beta=self.beta,
                temperature=3.0,
                warmup_factor=self.warmup_factor,
                DKD=self.dkd,
                **kwargs)
            loss_tckd_weak_2 = self.kd_loss_weight * (loss_tckd_weak_2 * mask).mean()
            loss_nckd_weak_2 = self.kd_loss_weight * (loss_nckd_weak_2 * mask).mean()
            loss_tckd_weak_3,loss_nckd_weak_3 = self.deepkd_plugin(
                logits_student_weak, 
                logits_teacher_weak, 
                target,
                alpha=self.alpha,
                beta=self.beta,
                temperature=5.0,
                warmup_factor=self.warmup_factor,
                DKD=self.dkd,
                **kwargs)
            loss_tckd_weak_3 = self.kd_loss_weight * (loss_tckd_weak_3 * mask).mean()
            loss_nckd_weak_3 = self.kd_loss_weight * (loss_nckd_weak_3 * mask).mean()
            loss_tckd_weak_4,loss_nckd_weak_4 = self.deepkd_plugin(
                logits_student_weak, 
                logits_teacher_weak, 
                target,
                alpha=self.alpha,
                beta=self.beta,
                temperature=2.0,
                warmup_factor=self.warmup_factor,
                DKD=self.dkd,
                **kwargs)
            loss_tckd_weak_4 = self.kd_loss_weight * (loss_tckd_weak_4 * mask).mean()
            loss_nckd_weak_4 = self.kd_loss_weight * (loss_nckd_weak_4 * mask).mean()
            loss_tckd_weak_5,loss_nckd_weak_5 = self.deepkd_plugin(
                logits_student_weak, 
                logits_teacher_weak, 
                target, 
                alpha=self.alpha,
                beta=self.beta,
                temperature=6.0,
                warmup_factor=self.warmup_factor,
                DKD=self.dkd,
                **kwargs)
            loss_tckd_weak_5 = self.kd_loss_weight * (loss_tckd_weak_5 * mask).mean()
            loss_nckd_weak_5 = self.kd_loss_weight * (loss_nckd_weak_5 * mask).mean()
            loss_tckd_weak = loss_tckd_weak_1 + loss_tckd_weak_2 + loss_tckd_weak_3 + loss_tckd_weak_4 + loss_tckd_weak_5
            loss_nckd_weak = loss_nckd_weak_1 + loss_nckd_weak_2 + loss_nckd_weak_3 + loss_nckd_weak_4 + loss_nckd_weak_5
            
            #strong
            loss_tckd_strong_1,loss_nckd_strong_1 = self.deepkd_plugin(
                logits_student_strong, 
                logits_teacher_strong, 
                target,
                alpha=self.alpha,
                beta=self.beta,
                temperature=self.temperature,
                warmup_factor=self.warmup_factor,
                DKD=self.dkd,
                **kwargs)
            loss_tckd_strong_1 = self.kd_loss_weight * loss_tckd_strong_1 
            loss_nckd_strong_1 = self.kd_loss_weight * loss_nckd_strong_1 
            loss_tckd_strong_2,loss_nckd_strong_2 = self.deepkd_plugin(
                logits_student_strong, 
                logits_teacher_strong, 
                target,
                alpha=self.alpha,
                beta=self.beta,
                temperature=3.0,
                warmup_factor=self.warmup_factor,
                DKD=self.dkd,
                **kwargs)
            loss_tckd_strong_2 = self.kd_loss_weight * loss_tckd_strong_2 
            loss_nckd_strong_2 = self.kd_loss_weight * loss_nckd_strong_2
            loss_tckd_strong_3,loss_nckd_strong_3 = self.deepkd_plugin(
                logits_student_strong, 
                logits_teacher_strong, 
                target,
                alpha=self.alpha,
                beta=self.beta,
                temperature=5.0,
                warmup_factor=self.warmup_factor,
                DKD=self.dkd,
                **kwargs)
            loss_tckd_strong_3 = self.kd_loss_weight * loss_tckd_strong_3 
            loss_nckd_strong_3 = self.kd_loss_weight * loss_nckd_strong_3
            loss_tckd_strong_4,loss_nckd_strong_4 = self.deepkd_plugin(
                logits_student_strong, 
                logits_teacher_strong, 
                target,
                alpha=self.alpha,
                beta=self.beta,
                temperature=2.0,
                warmup_factor=self.warmup_factor,
                DKD=self.dkd,
                **kwargs)
            loss_tckd_strong_4 = self.kd_loss_weight * loss_tckd_strong_4 
            loss_nckd_strong_4 = self.kd_loss_weight * loss_nckd_strong_4
            loss_tckd_strong_5,loss_nckd_strong_5 = self.deepkd_plugin(
                logits_student_strong, 
                logits_teacher_strong, 
                target,
                alpha=self.alpha,
                beta=self.beta,
                temperature=6.0,
                warmup_factor=self.warmup_factor,
                DKD=self.dkd,
                **kwargs)
            loss_tckd_strong_5 = self.kd_loss_weight * loss_tckd_strong_5 
            loss_nckd_strong_5 = self.kd_loss_weight * loss_nckd_strong_5 
            loss_tckd_strong = loss_tckd_strong_1 + loss_tckd_strong_2 + loss_tckd_strong_3 + loss_tckd_strong_4 + loss_tckd_strong_5
            loss_nckd_strong = loss_nckd_strong_1 + loss_nckd_strong_2 + loss_nckd_strong_3 + loss_nckd_strong_4 + loss_nckd_strong_5
            
            
            
            loss_cc_weak = self.kd_loss_weight * ((cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                self.temperature,
                # reduce=False
            ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                3.0,
            ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                5.0,
            ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                2.0,
            ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                6.0,
            ) * class_conf_mask).mean())
            loss_cc_strong = self.kd_loss_weight * cc_loss(
                logits_student_strong,
                logits_teacher_strong,
                self.temperature,
            ) + self.kd_loss_weight * cc_loss(
                logits_student_strong,
                logits_teacher_strong,
                3.0,
            ) + self.kd_loss_weight * cc_loss(
                logits_student_strong,
                logits_teacher_strong,
                5.0,
            ) + self.kd_loss_weight * cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                2.0,
            ) + self.kd_loss_weight * cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                6.0,
            )
            loss_bc_weak = self.kd_loss_weight * ((bc_loss(
                logits_student_weak,
                logits_teacher_weak,
                self.temperature,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_weak,
                logits_teacher_weak,
                3.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_weak,
                logits_teacher_weak,
                5.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_weak,
                logits_teacher_weak,
                2.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_weak,
                logits_teacher_weak,
                6.0,
            ) * mask).mean())
            loss_bc_strong = self.kd_loss_weight * ((bc_loss(
                logits_student_strong,
                logits_teacher_strong,
                self.temperature,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_strong,
                logits_teacher_strong,
                3.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_strong,
                logits_teacher_strong,
                5.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_strong,
                logits_teacher_strong,
                2.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_strong,
                logits_teacher_strong,
                6.0,
            ) * mask).mean())
            losses_dict = {
                "loss_ce": loss_ce+loss_cc_weak+loss_bc_weak,
                "loss_tckd": loss_tckd_weak+loss_tckd_strong,
                "loss_nckd": loss_nckd_weak+loss_nckd_strong,
            }
            return logits_student_weak, losses_dict
        
        else:
            # losses
            loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))
            loss_kd_weak = self.kd_loss_weight * ((kd_loss(
                logits_student_weak,
                logits_teacher_weak,
                self.temperature,
                # reduce=False
                logit_stand=self.logit_stand,
            ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
                logits_student_weak,
                logits_teacher_weak,
                3.0,
                # reduce=False
                logit_stand=self.logit_stand,
            ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
                logits_student_weak,
                logits_teacher_weak,
                5.0,
                # reduce=False
                logit_stand=self.logit_stand,
            ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
                logits_student_weak,
                logits_teacher_weak,
                2.0,
                # reduce=False
                logit_stand=self.logit_stand,
            ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
                logits_student_weak,
                logits_teacher_weak,
                6.0,
                # reduce=False
                logit_stand=self.logit_stand,
            ) * mask).mean())

            loss_kd_strong = self.kd_loss_weight * kd_loss(
                logits_student_strong,
                logits_teacher_strong,
                self.temperature,
                logit_stand=self.logit_stand,
            ) + self.kd_loss_weight * kd_loss(
                logits_student_strong,
                logits_teacher_strong,
                3.0,
                logit_stand=self.logit_stand,
            ) + self.kd_loss_weight * kd_loss(
                logits_student_strong,
                logits_teacher_strong,
                5.0,
                logit_stand=self.logit_stand,
            ) + self.kd_loss_weight * kd_loss(
                logits_student_weak,
                logits_teacher_weak,
                2.0,
                logit_stand=self.logit_stand,
            ) + self.kd_loss_weight * kd_loss(
                logits_student_weak,
                logits_teacher_weak,
                6.0,
                logit_stand=self.logit_stand,
            )

            loss_cc_weak = self.kd_loss_weight * ((cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                self.temperature,
                # reduce=False
            ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                3.0,
            ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                5.0,
            ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                2.0,
            ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                6.0,
            ) * class_conf_mask).mean())
            loss_cc_strong = self.kd_loss_weight * cc_loss(
                logits_student_strong,
                logits_teacher_strong,
                self.temperature,
            ) + self.kd_loss_weight * cc_loss(
                logits_student_strong,
                logits_teacher_strong,
                3.0,
            ) + self.kd_loss_weight * cc_loss(
                logits_student_strong,
                logits_teacher_strong,
                5.0,
            ) + self.kd_loss_weight * cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                2.0,
            ) + self.kd_loss_weight * cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                6.0,
            )
            loss_bc_weak = self.kd_loss_weight * ((bc_loss(
                logits_student_weak,
                logits_teacher_weak,
                self.temperature,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_weak,
                logits_teacher_weak,
                3.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_weak,
                logits_teacher_weak,
                5.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_weak,
                logits_teacher_weak,
                2.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_weak,
                logits_teacher_weak,
                6.0,
            ) * mask).mean())
            loss_bc_strong = self.kd_loss_weight * ((bc_loss(
                logits_student_strong,
                logits_teacher_strong,
                self.temperature,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_strong,
                logits_teacher_strong,
                3.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_strong,
                logits_teacher_strong,
                5.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_strong,
                logits_teacher_strong,
                2.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_strong,
                logits_teacher_strong,
                6.0,
            ) * mask).mean())
            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_kd_weak + loss_kd_strong,
                "loss_cc": loss_cc_weak,
                "loss_bc": loss_bc_weak
            }
            return logits_student_weak, losses_dict

