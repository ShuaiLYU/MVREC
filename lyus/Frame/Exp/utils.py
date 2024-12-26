
__all__ = ["SingletonType","get_folder_name_of_file","get_current_time_point","fix_seeds","save_to_txt"]

import  os
import numpy as np
"""

 a experiment save_root:

 save_root + "project_name" +"exp_name"+"run_name"



"""


import threading

"""
https://www.cnblogs.com/huchong/p/8244279.html
"""
class SingletonType(type):
    _instance_lock = threading.Lock()
    def __call__(cls, *args, **kwargs):            

        # print("new",kwargs.get("new",False))
        if  hasattr(cls, "_instance") and kwargs.get("new",False):
            with SingletonType._instance_lock:   
                # print(cls._instance)
                # print(kwargs)
                cls._instance=None
                cls._instance = super(SingletonType,cls).__call__(*args, **kwargs)
                
        if not hasattr(cls, "_instance"): 
            with SingletonType._instance_lock:                  
                if not hasattr(cls, "_instance"):
                    cls._instance = super(SingletonType,cls).__call__(*args, **kwargs)
        return cls._instance





import time
def get_current_time_point():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())



"""
get the folder name in which  the file is located
获得输入文件名，所在文件夹的名字

"""
def get_folder_name_of_file(file):
    dir_name=os.path.split(os.path.realpath(file))[0]
    return os.path.basename(dir_name)




def is_scalar(value):
    try:
        value =float(value)
    except:
        return False
    return True

def check_scalar(inp):
    
    try:
        value = float(inp)
        if value.is_integer():
            value = int(value)
        return value
    except:
        return inp





import random 
import numpy as np
import torch
def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
def save_to_txt(save_dir,filename,stri):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir,filename), 'w') as f:
        f.write(stri)
        
        
        
        
