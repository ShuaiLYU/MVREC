import logging
import os
from .param import Param
from .utils import *
from .arg import *
from .logger import get_logger
from .tb_writer import TbWriter
from .visualizer import Visualizer

__all__ = ["Experiment", "run_an_exper", "run_group_exper","build_new_exper"]

import time

from .utils import  is_scalar

class Experiment(metaclass=SingletonType):
    def __init__(self, save_root=None, project_name=None, exp_name=None,
                 run_name=None, seed=None, **kwargs):
        self.save_dir = save_root
        self.project_name = project_name
        self.exp_name = exp_name
        self.run_name = run_name
        if self.project_name is not None:
            self.save_dir = os.path.join(self.save_dir, self.project_name)
        if self.exp_name is not None:
            self.save_dir = os.path.join(self.save_dir, self.exp_name)
        if self.run_name is not None:
            if self.run_name == "runtime":
                self.run_name = "runtime_" + get_current_time_point()
            self.save_dir =  os.path.join(self.save_dir, self.run_name )
        self._meta_data = {}
        # #创建单例，log模块
        # self.init_global_logger(os.path.join(self.save_dir,"log.txt"))

        if seed is not None:
            fix_seeds(seed)

        self._meta_data["seed"] = seed

    def set_attribute(self, key, val):
        self._meta_data[key] = val

    def set_attr(self, key, val):
        self._meta_data[key] = val

    def get(self, key):
        return self._meta_data[key]

    def set_arg(self, arg: dict):
        self._arg = arg
        save_to_txt(self.save_dir, "arg", str(self._arg))

    def set_param(self, param: Param):
        self._param = param
        save_to_txt(self.save_dir, "param", str(self._param))

    def get_param(self) -> Param:
        return self._param

    def get_save_dir(self):
        return self.save_dir

    ##########################################################################
    def get_logger(self):
        if not hasattr(self, "_logger"):
            self.set_logger()
        return self._logger

    def set_logger(self, filename="log"):
        self._logger = get_logger(self.save_dir, filename)

    def info(self, string):
        if not hasattr(self, "_logger"):
            self.set_logger()
        self._logger.info(string)

    ##########################################################################
    def set_writer(self, folder="tblog"):
        log_dir = os.path.join(self.save_dir, folder)
        self._writer = TbWriter(log_dir)

    def get_writer(self):
        if not hasattr(self, "_writer"):
            self.set_writer()
        return self._writer

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if not hasattr(self, "_writer"):
            self.set_writer()
        self._writer.add_scalar(tag, scalar_value, global_step, walltime)
        
    def add_train_var(self,data:dict):
        if not hasattr(self, "_train_epo_log"):
            from lyus.FileTool import CsvLogger
            self._train_epo_log=CsvLogger(self.save_dir,"train_epo_log")
        data = { k:v for k,v in data.items() if is_scalar(v)}
        if len(data)>0:
            self._train_epo_log.append_one_row(data)
    ##########################################################################

def build_new_exper(param_name, param, save_root, project_name,exp_name):
    exper = Experiment(save_root=save_root, project_name=project_name,
                       exp_name=exp_name, run_name=param_name, new=True)

    exper.set_param(param)
    # exper.set_arg(arg_tool)
    exper.set_logger()
    return exper

def run_an_exper(param_name, param, save_root, project_name, run_an_exper,run_name=None):
    exper = Experiment(save_root=save_root, project_name=project_name,
                       exp_name=param_name, run_name=run_name, new=True)
    assert False # disable
    exper.set_param(param)
    # exper.set_arg(arg_tool)
    exper.set_logger()
    run_an_exper(exper)  # 一组参数的运行函数
    del exper


def run_group_exper(param_group, save_root, project_name, run_an_exper):
    for idx, param_name, param in enumerate(param_group):
        run_an_exper(param_name, param, save_root, project_name, run_an_exper)






