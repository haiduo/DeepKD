import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller

def kd_loss(logits_student, logits_teacher, temperature, reduce=False):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd

class CRLD(Distiller):
    """Cross-View Consistency Regularisation for Knowledge Distillation (ACMMM2024)"""

    def __init__(self, student, teacher, cfg):
        super(CRLD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.CRLD.CE_WEIGHT
        self.wv_loss_weight = cfg.CRLD.WV_WEIGHT
        self.cv_loss_weight = cfg.CRLD.CV_WEIGHT
        self.t = cfg.CRLD.TEMPERATURE
        self.tau_w = cfg.CRLD.TAU_W
        self.tau_s = cfg.CRLD.TAU_S
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND 
        self.trainer = cfg.SOLVER.TRAINER
        #deepkd
        self.alpha = cfg.SOLVER.DEEPKD.ALPHA
        self.beta = cfg.SOLVER.DEEPKD.BETA
        self.warmup_factor = cfg.SOLVER.DEEPKD.WARMUP_FACTOR
        self.dkd = cfg.SOLVER.DEEPKD.DKD
        self.topk = cfg.SOLVER.DEEPKD.TOPK

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)

        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)
            # logits_teacher_w = self.teacher(image_w) # for vit teacher
            # logits_teacher_s = self.teacher(image_s) # for vit teacher

        pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
        conf_weak, _ = pred_teacher_weak.max(dim=1)
        conf_weak = conf_weak.detach()
        mask_weak = conf_weak.ge(self.tau_w).bool()

        pred_teacher_strong = F.softmax(logits_teacher_strong.detach(), dim=1)
        conf_strong, _ = pred_teacher_strong.max(dim=1)
        conf_strong = conf_strong.detach()
        mask_strong = conf_strong.ge(self.tau_s).bool()

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))
        if self.trainer == 'deepkd' or self.trainer == 'deepkd_aug':
            loss_tckd_wv_1, loss_nckd_wv_1 = self.deepkd_plugin(logits_student_weak, 
                logits_teacher_weak.detach(), 
                target,
                alpha=self.alpha,
                beta=self.beta,
                temperature=self.t,
                warmup_factor=self.warmup_factor,
                DKD=self.dkd,
                topk=self.topk,
                **kwargs)
            loss_tckd_wv_2, loss_nckd_wv_2 = self.deepkd_plugin(logits_student_strong, 
                logits_teacher_strong.detach(), 
                target,
                alpha=self.alpha,
                beta=self.beta,
                temperature=self.t,
                warmup_factor=self.warmup_factor,
                DKD=self.dkd,
                topk=self.topk,
                **kwargs)
            
            loss_tckd_wv = loss_tckd_wv_1 + loss_tckd_wv_2
            loss_nckd_wv = loss_nckd_wv_1 + loss_nckd_wv_2
            
            loss_nckd_wv = self.wv_loss_weight * (loss_nckd_wv * mask_weak).mean()
            loss_tckd_wv = self.wv_loss_weight * (loss_tckd_wv * mask_weak).mean()
            
            loss_tckd_cv_1, loss_nckd_cv_1 = self.deepkd_plugin(logits_student_strong, 
                logits_teacher_weak.detach(), 
                target,
                alpha=self.alpha,
                beta=self.beta,
                temperature=self.t,
                warmup_factor=self.warmup_factor,
                DKD=self.dkd,
                topk=self.topk,
                **kwargs)
            loss_tckd_cv_2, loss_nckd_cv_2 = self.deepkd_plugin(logits_student_weak, 
                logits_teacher_strong.detach(), 
                target,
                alpha=self.alpha,
                beta=self.beta,
                temperature=self.t,
                warmup_factor=self.warmup_factor,
                DKD=self.dkd,
                topk=self.topk,
                **kwargs)
            
            loss_tckd_cv = loss_tckd_cv_1 + loss_tckd_cv_2
            loss_nckd_cv = loss_nckd_cv_1 + loss_nckd_cv_2
            
            loss_nckd_cv = self.cv_loss_weight * (loss_nckd_cv * mask_strong).mean()
            loss_tckd_cv = self.cv_loss_weight * (loss_tckd_cv * mask_strong).mean()

            losses_dict = {
            "loss_ce": loss_ce,
            "loss_nckd": loss_nckd_wv+loss_nckd_cv,
            "loss_tckd": loss_tckd_wv+loss_tckd_cv,
            }

            return logits_student_weak, losses_dict
        # CRLD losses
        loss_kd_wv = self.wv_loss_weight * ((kd_loss(logits_student_weak, logits_teacher_weak.detach(), self.t)
                                             + kd_loss(logits_student_strong, logits_teacher_strong.detach(), self.t)) * mask_weak).mean()
        loss_kd_cv = self.cv_loss_weight * ((kd_loss(logits_student_strong, logits_teacher_weak.detach(), self.t)
                                             + kd_loss(logits_student_weak, logits_teacher_strong.detach(), self.t)) * mask_strong).mean()

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd_wv+loss_kd_cv,
        }

        return logits_student_weak, losses_dict