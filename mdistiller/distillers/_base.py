import torch
import torch.nn as nn
import torch.nn.functional as F


class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self._last_student_logits = None
        self._last_teacher_logits = None

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def get_logits(self):
        """
        Returns:
            tuple: (student_logits, teacher_logits)
        """
        return self._last_student_logits, self._last_teacher_logits

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])
    def normalize(self,logit):
        mean = logit.mean(dim=-1, keepdims=True)
        stdv = logit.std(dim=-1, keepdims=True)
        return (logit - mean) / (1e-7 + stdv)
    
    def deepkd_plugin(self,logits_student_in, logits_teacher_in, target, alpha=1, beta=8, temperature=4,warmup_factor=20,DKD=False,logit_stand=False,**kwargs):

        logits_student = self.normalize(logits_student_in) if logit_stand else logits_student_in
        logits_teacher = self.normalize(logits_teacher_in) if logit_stand else logits_teacher_in

        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = self.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (temperature**2)
            / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (temperature**2)
            / target.shape[0]
        )
        
        if DKD:
            warmup_factor = min(kwargs["epoch"] / warmup_factor, 1.0)
            loss_tckd = warmup_factor * tckd_loss
            loss_nckd = warmup_factor * nckd_loss
            return alpha * loss_tckd, beta * loss_nckd
        else:
            pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
            teacher_target_class_prob = pred_teacher.gather(1, target.unsqueeze(1))  # 获取教师目标类的概率
            nckd_loss = (1 - teacher_target_class_prob) * nckd_loss
            return tckd_loss, nckd_loss



    def deepkd_plugin(self,logits_student_in, logits_teacher_in, target, alpha=1, beta=8, temperature=4,warmup_factor=20,DKD=False,topk=0,logit_stand=False,**kwargs):
        logits_student = self.normalize(logits_student_in) if logit_stand else logits_student_in
        logits_teacher = self.normalize(logits_teacher_in) if logit_stand else logits_teacher_in

        gt_mask = self._get_gt_mask(logits_student, target) 
        topk_mask = self._get_topk_mask(logits_teacher, target,topk) 
        other_mask = self._get_other_mask(logits_student, target) 

        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = self.cat_mask(pred_student, gt_mask , other_mask)
        pred_teacher = self.cat_mask(pred_teacher, gt_mask , other_mask)


        log_pred_student = torch.log(pred_student)
        # Calculate TCKD loss:
        tckd_loss = (F.kl_div(log_pred_student, pred_teacher, size_average=False)* (temperature**2)/target.shape[0])

        # Calculatel NCKD loss:
        pred_teacher_part2 = F.softmax(logits_teacher / temperature - 1000.0 * gt_mask, dim=1)
        log_pred_student_part2 = F.log_softmax(logits_student / temperature - 1000.0 * gt_mask, dim=1)
        nckd_loss = (F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)* (temperature**2)/ target.shape[0])

        # Calculate NCKD loss:
        pred_teacher_part3 = F.softmax(logits_teacher / temperature - 1000.0 * topk_mask, dim=1)
        log_pred_student_part3 = F.log_softmax(logits_student / temperature - 1000.0 * topk_mask, dim=1)
      
        topk_nckd_loss = (F.kl_div(log_pred_student_part3, pred_teacher_part3, size_average=False)* (temperature**2)/ target.shape[0])
        if topk :
            if DKD:
                warmup_factor = min(kwargs["epoch"] / warmup_factor, 1.0)
                loss_tckd = warmup_factor * tckd_loss
                topk_nckd_loss = warmup_factor * topk_nckd_loss
                return alpha * loss_tckd, beta * topk_nckd_loss
            else:
                pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
                teacher_target_class_prob = pred_teacher.gather(1, target.unsqueeze(1))  
                topk_nckd_loss = (1 - teacher_target_class_prob) * topk_nckd_loss
                return tckd_loss, topk_nckd_loss
        else:
            if DKD:
                warmup_factor = min(kwargs["epoch"] / warmup_factor, 1.0)
                loss_tckd = warmup_factor * tckd_loss
                loss_nckd = warmup_factor * nckd_loss
                return alpha * loss_tckd, beta * loss_nckd
            else:
                pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
                teacher_target_class_prob = pred_teacher.gather(1, target.unsqueeze(1))  
                nckd_loss = (1 - teacher_target_class_prob) * nckd_loss
                return tckd_loss, nckd_loss


    def _get_topk_mask(self,logits, target, top_k=20):
        target = target.reshape(-1)

        mask = torch.ones_like(logits)
        
        mask.scatter_(1, target.unsqueeze(1), -1000) 
        logits_after_mask = logits.reshape(logits.size(0), -1)
        _, top_indices = torch.topk(logits_after_mask, k=min(top_k, logits_after_mask.size(1)), dim=1)
        
        batch_indices = torch.arange(logits.size(0)).unsqueeze(1).expand_as(top_indices)
        
        mask = torch.ones_like(logits)
        mask[batch_indices, top_indices] = 0
        
        mask.scatter_(1, target.unsqueeze(1), 1)
        
        return mask.bool()
    
    def _get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask
    
    def cat_mask(self,t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt


class Vanilla(nn.Module):
    def __init__(self, student):
        super(Vanilla, self).__init__()
        self.student = student

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        loss = F.cross_entropy(logits_student, target)
        return logits_student, {"ce": loss}

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)[0]
