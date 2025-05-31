"""
Logger module for EmotionNet.
Provides consistent logging setup across the project.
"""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name, log_level=logging.INFO, log_file=None):
    """
    Set up logger with consistent formatting and optional file output.
    
    Args:
        name: Name of the logger
        log_level: Logging level (default: INFO)
        log_file: Path to log file (optional)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_run_log_file(log_dir='logs', prefix='run'):
    """
    Generate a timestamped log file path.
    
    Args:
        log_dir: Directory for log files (default: 'logs')
        prefix: Log file name prefix (default: 'run')
        
    Returns:
        Path to log file
    """
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped file name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{prefix}_{timestamp}.log"
    
    return str(log_file)


class TrainingLogger:
    """
    Specialized logger for training processes with progress tracking.
    """
    def __init__(self, name='training', log_file=None):
        self.logger = setup_logger(name, log_file=log_file)
        self.epoch = 0
        self.total_epochs = 0
        self.step = 0
        self.total_steps = 0
    
    def set_epoch_progress(self, epoch, total_epochs):
        """Set current epoch and total epochs"""
        self.epoch = epoch
        self.total_epochs = total_epochs
    
    def set_step_progress(self, step, total_steps):
        """Set current step and total steps"""
        self.step = step
        self.total_steps = total_steps
    
    def log_epoch_start(self, epoch, total_epochs):
        """Log start of an epoch"""
        self.set_epoch_progress(epoch, total_epochs)
        self.logger.info(f"Epoch {epoch}/{total_epochs} started")
    
    def log_epoch_end(self, epoch_metrics):
        """Log end of an epoch with metrics"""
        metrics_str = " ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()])
        self.logger.info(f"Epoch {self.epoch}/{self.total_epochs} completed | {metrics_str}")
    
    def log_step(self, step_metrics, step=None, total_steps=None):
        """Log training step with metrics"""
        if step is not None and total_steps is not None:
            self.set_step_progress(step, total_steps)
        
        if step_metrics:
            metrics_str = " ".join([f"{k}: {v:.4f}" for k, v in step_metrics.items()])
            self.logger.info(f"Epoch {self.epoch}/{self.total_epochs} | "
                           f"Step {self.step}/{self.total_steps} | {metrics_str}")
    
    def log_validation(self, val_metrics):
        """Log validation results"""
        if val_metrics:
            metrics_str = " ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            self.logger.info(f"Validation | {metrics_str}")
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)


# For backwards compatibility
get_logger = setup_logger 