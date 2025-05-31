"""
Sharpness Aware Minimization (SAM) optimizer
Helps find flatter minima for better generalization
"""

import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    """
    Sharpness Aware Minimization optimizer
    Improves generalization by seeking flatter minima
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, **kwargs):
        assert isinstance(base_optimizer, type)
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class SAMOptimizer(Optimizer):
    """
    Wrapper class for SAM optimizer to maintain compatibility with the training script.
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, **kwargs):
        """
        Initialize SAMOptimizer wrapper around existing base optimizer instance.
        
        Args:
            params (iterable): Model parameters to optimize
            base_optimizer: Base optimizer instance (not class type)
            rho (float): Size of the perturbation for SAM, default 0.05
            adaptive (bool): Use adaptive SAM, default True
        """
        self.param_groups = []
        self._params = list(params)
        
        # Extract parameters from base optimizer if it's already instantiated
        if not isinstance(base_optimizer, type):
            # Store the actual base optimizer instance
            self.base_optimizer = base_optimizer
            defaults = dict(rho=rho, adaptive=adaptive)
            super(SAMOptimizer, self).__init__(self._params, defaults)
            
            # Copy param groups from base optimizer
            self.param_groups = self.base_optimizer.param_groups
            if hasattr(self.base_optimizer, 'defaults'):
                self.defaults.update(self.base_optimizer.defaults)
        else:
            # If base_optimizer is a class, create an instance (same as SAM behavior)
            defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
            super(SAMOptimizer, self).__init__(self._params, defaults)
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
            self.param_groups = self.base_optimizer.param_groups
            self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, first_step=True):
        """
        Compute and save the perturbation for SAM
        
        Args:
            first_step (bool): Flag to indicate this is the first step, always True for this method
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group.get("rho", 0.05) / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                adaptive = group.get("adaptive", True)
                e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                
                # Save the perturbation
                if not hasattr(self, "state"):
                    self.state = {}
                if p not in self.state:
                    self.state[p] = {}
                self.state[p]["e_w"] = e_w
        
        return True  # Return True to match the SAM behavior
    
    @torch.no_grad()
    def second_step(self, second_step=True):
        """
        Apply the SAM update by first removing the perturbation and then applying the optimizer step
        
        Args:
            second_step (bool): Flag to indicate this is the second step, always True for this method
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p not in self.state: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        return True  # Return True to match expected behavior
    
    def step(self, closure=None):
        """
        Performs a single optimization step using SAM.
        
        Args:
            closure (callable): A closure that re-evaluates the model and returns the loss
        """
        if closure is not None:
            with torch.enable_grad():
                closure()
        return self.base_optimizer.step()
    
    def zero_grad(self, set_to_none=False):
        """Zero out the gradients"""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
    
    def _grad_norm(self):
        """Calculate the gradient norm for all parameters"""
        if not self.param_groups or not self.param_groups[0]["params"]:
            return torch.tensor(0.0)
            
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group.get("adaptive", True) else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict"""
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups 