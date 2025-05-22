import math
import torch
from torch import Tensor
import torch.optim._functional as F
from torch.optim.optimizer import Optimizer, required
from typing import List, Optional


def check_in(t, l):
    for i in l:
        if t is i:
            return True
    return False


def deepkd(params: List[Tensor],
         d_p_list: List[Tensor],
         momentum_buffer_list: List[Optional[Tensor]],
         tcg_grad_buffer: List[Optional[Tensor]],
         tcg_momentum_buffer: List[Optional[Tensor]],
         tcg_params: List[Tensor],
         ncg_grad_buffer: List[Optional[Tensor]],
         ncg_momentum_buffer: List[Optional[Tensor]],
         ncg_params: List[Tensor],
         *,
         weight_decay: float,
         momentum_tog: float,
         momentum_tcg: float,
         momentum_ncg: float,
         lr: float,
         dampening: float):
    """
    deepkd optimization function, handling three different gradients and momenta: TOG, TCG, and NCG
    """
    # handle TOG 
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if momentum_tog != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            elif check_in(param, tcg_params):             
                buf.mul_(momentum_tcg).add_(d_p, alpha=1 - dampening)
            elif check_in(param, ncg_params):                
                buf.mul_(momentum_ncg).add_(d_p, alpha=1 - dampening)
            else:
                buf.mul_((momentum_tcg+momentum_ncg + momentum_tog) / 3.).add_(d_p, alpha=1 - dampening)
            d_p = buf
        # update parameter
        param.add_(d_p, alpha=-lr)


    # handle NCG 
    for i, (d_p, buf, p) in enumerate(zip(ncg_grad_buffer, ncg_momentum_buffer, ncg_params)):
        if buf is None:
            buf = torch.clone(d_p).detach()
            ncg_momentum_buffer[i] = buf
        elif check_in(p, params):
            buf.mul_(momentum_ncg).add_(d_p, alpha=1 - dampening)
        elif check_in(p, tcg_params):
            buf.mul_(momentum_tcg).add_(d_p, alpha=1 - dampening)
        else:
            if weight_decay != 0:
                d_p = d_p.add(p, alpha=weight_decay)
            buf.mul_((momentum_tcg+momentum_ncg + momentum_tog) / 3.).add_(d_p, alpha=1 - dampening)
        # update parameter
        p.add_(buf, alpha=-lr)

    # handle TCG
    for i, (d_p, buf, p) in enumerate(zip(tcg_grad_buffer, tcg_momentum_buffer, tcg_params)):
        if weight_decay != 0:
            d_p = d_p.add(p, alpha=weight_decay)
        if buf is None:
            buf = torch.clone(d_p).detach()
            tcg_momentum_buffer[i] = buf
        elif check_in(p, params):
            buf.mul_(momentum_tcg).add_(d_p, alpha=1 - dampening)
        elif check_in(p, ncg_params):
            buf.mul_(momentum_ncg).add_(d_p, alpha=1 - dampening)
        else:
            if weight_decay != 0:
               d_p = d_p.add(p, alpha=weight_decay)
            buf.mul_((momentum_tcg+momentum_ncg + momentum_tog) / 3.).add_(d_p, alpha=1 - dampening)   
        # update parameter
        p.add_(buf, alpha=-lr)


class DEEPKDTrainer(Optimizer):


    def __init__(
        self, 
        params, 
        lr=required, 
        momentum_tog=0,           # task loss momentum
        momentum_tcg=0,      # TCKD loss momentum
        momentum_ncg=0,      # NCKD loss momentum
        dampening=0,
        weight_decay=0):
            
        defaults = dict(
            lr=lr, 
            momentum_tog=momentum_tog, 
            momentum_tcg=momentum_tcg,
            momentum_ncg=momentum_ncg,
            dampening=dampening,
            weight_decay=weight_decay
        )
        
        # Initialize buffers
        self.tcg_grad_buffer = []
        self.tcg_params = []
        self.tcg_momentum_buffer = []
        
        self.ncg_grad_buffer = []
        self.ncg_params = []
        self.ncg_momentum_buffer = []
        
        super(DEEPKDTrainer, self).__init__(params, defaults)

    @torch.no_grad()
    def step_tcg(self, closure=None):
        """Collect TCG information"""    
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        assert len(self.param_groups) == 1, "Only implement for one-group params."
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_tcg_buffer_list = []
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if 'momentum_tcg_buffer' not in state:
                        momentum_tcg_buffer_list.append(None)
                    else:
                        momentum_tcg_buffer_list.append(state['momentum_tcg_buffer'])
                    
        self.tcg_momentum_buffer = momentum_tcg_buffer_list
        self.tcg_grad_buffer = d_p_list
        self.tcg_params = params_with_grad
        return loss

    @torch.no_grad()
    def step_ncg(self, closure=None):
        """Collect NCG information"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        assert len(self.param_groups) == 1, "Only implement for one-group params."
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_ncg_buffer_list = []
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if 'momentum_ncg_buffer' not in state:
                        momentum_ncg_buffer_list.append(None)
                    else:
                        momentum_ncg_buffer_list.append(state['momentum_ncg_buffer'])
                    
        self.ncg_momentum_buffer = momentum_ncg_buffer_list
        self.ncg_grad_buffer = d_p_list
        self.ncg_params = params_with_grad
        return loss

    @torch.no_grad()
    def step(self, closure=None):
        """Execute parameter update"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        assert len(self.param_groups) == 1, "Only implement for one-group params."
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum_tog = group['momentum_tog']
            momentum_tcg = group['momentum_tcg']
            momentum_ncg = group['momentum_ncg']
            dampening = group['dampening']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
                        
            # Call deepkd optimization function
            deepkd(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                self.tcg_grad_buffer,
                self.tcg_momentum_buffer,
                self.tcg_params,
                self.ncg_grad_buffer,
                self.ncg_momentum_buffer,
                self.ncg_params,
                weight_decay=weight_decay,
                momentum_tog=momentum_tog,
                momentum_tcg=momentum_tcg,
                momentum_ncg=momentum_ncg,
                lr=lr,
                dampening=dampening
            )
            
            # Update momentum buffer in state dictionary
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
                
            for p, momentum_tcg_buffer in zip(self.tcg_params, self.tcg_momentum_buffer):
                state = self.state[p]
                state['momentum_tcg_buffer'] = momentum_tcg_buffer
                
            for p, momentum_ncg_buffer in zip(self.ncg_params, self.ncg_momentum_buffer):
                state = self.state[p]
                state['momentum_ncg_buffer'] = momentum_ncg_buffer
                
            # Clear buffers
            self.tcg_grad_buffer = []
            self.tcg_params = []
            self.tcg_momentum_buffer = []
            
            self.ncg_grad_buffer = []
            self.ncg_params = []
            self.ncg_momentum_buffer = []
            
        return loss 
