


import torch
from torch.utils.tensorboard import SummaryWriter

__all__ = ["TbWriter"]


from .utils import  is_scalar



import os
class TbWriter(object):
    
    def __init__(self,log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self._writer=SummaryWriter(log_dir)
        
        
    def add_scalar(self,tag, scalar_value, global_step=None, walltime=None):
        if(not is_scalar(scalar_value)):
            # print("[warning], {} check scalar failed!, discard... ".format(type(scalar_value)))
            return 
        self._writer.add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime)