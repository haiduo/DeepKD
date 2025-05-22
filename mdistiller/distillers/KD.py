import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND 
        self.trainer = cfg.SOLVER.TRAINER

        #deepkd-dkd
        self.alpha = cfg.SOLVER.DEEPKD.ALPHA
        self.beta = cfg.SOLVER.DEEPKD.BETA
        self.warmup_factor = cfg.SOLVER.DEEPKD.WARMUP_FACTOR
        self.dkd = cfg.SOLVER.DEEPKD.DKD
        self.topk= cfg.SOLVER.DEEPKD.TOPK
       
    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        if self.trainer == 'deepkd' or self.trainer == 'deepkd_aug':
            loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
            loss_tckd,loss_nckd = self.deepkd_plugin(
                logits_student, 
                logits_teacher, 
                target,
                alpha=self.alpha,
                beta=self.beta,
                temperature=self.temperature,
                warmup_factor=self.warmup_factor,
                DKD=self.dkd,
                logit_stand=self.logit_stand,
                ttopk=self.topk,
                **kwargs)
            losses_dict = {
                "loss_task": loss_ce,
                "loss_tckd": loss_tckd,
                "loss_nckd": loss_nckd,
            }
            return logits_student, losses_dict
        else:
        
            # losses
            loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
            loss_kd = self.kd_loss_weight * kd_loss(
                logits_student, logits_teacher, self.temperature, self.logit_stand
            )
            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_kd,
            }
            return logits_student, losses_dict
