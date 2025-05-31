"""
Optimizer configuration and setup for EmotionNet training.
"""

import torch.optim as optim
from .sam_optimizer import SAM
from ..config.base import Config


def create_optimizer(model, config: Config):
    """Create optimizer based on configuration."""
    # Check if SGD optimizer is specified in configuration
    use_sgd = (hasattr(config, 'optimizer') and 
               hasattr(config.optimizer, 'name') and 
               config.optimizer.name == 'sgd')
    
    if use_sgd:
        print(f"   ✅ Using SGD optimizer")
        return optim.SGD(
            model.parameters(),
            lr=config.training.lr,
            momentum=0.9,
            weight_decay=config.training.weight_decay
        )
    else:
        # Default: SAM optimizer for complex training
        print(f"   ✅ Using SAM optimizer (default)")
    return SAM(
        model.parameters(),
        optim.AdamW,
        rho=config.training.sam_rho,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )


def create_scheduler(optimizer, config: Config):
    """Create scheduler based on configuration."""
    # Check if Plateau scheduler is specified in configuration
    use_plateau = (hasattr(config, 'scheduler') and 
                   hasattr(config.scheduler, 'name') and 
                   config.scheduler.name == 'plateau')
    
    if use_plateau:
        print(f"   ✅ Using ReduceLROnPlateau scheduler")
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        # Default: OneCycleLR for complex training
        print(f"   ✅ Using OneCycleLR scheduler (default)")
    total_steps = config.training.epochs * getattr(config.training, 'steps_per_epoch', 100)
    return optim.lr_scheduler.OneCycleLR(
        optimizer.base_optimizer,
        max_lr=config.training.max_lr,
        total_steps=total_steps,
        pct_start=config.training.pct_start,
        anneal_strategy=config.training.anneal_strategy,
        div_factor=config.training.div_factor,
        final_div_factor=config.training.final_div_factor
    ) 