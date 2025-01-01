

import os,sys
# p= r"/home/lyushuai/Projects/wise_pro/FabricDataSolution/"
# assert os.path.exists( os.path.join(p,"FabricDataLib" ))
# sys.path.append(p)


from .clip_preprocess import ClipPreprocess
from .model import ClipModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18,resnet50,vgg11,vgg11_bn


import numpy as np
import torch 
import torch.nn as nn
import os
import random
import pandas as pd

from torchvision import transforms
from PIL import Image
import lyus.data.my_transforms as my_trans
import lyus.Frame as FM


def print_trainable_parameters(model):
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}, shape: {param.shape}")


def create_model(exper:FM.Experiment):
    PARAM=exper.get_param()


    text_list=PARAM.data.class_names
    input_shape=PARAM.data.input_shape
    # print(text_list,1111)
    model= ClipModel(**PARAM.ClipModel)
    DEVICE="cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    model.set_trainable_attention_params()
    print_trainable_parameters(model)
    # print(model)
    


    return model




from .multi_view_data import MultiviewRoiData


def create_train_data(exper:FM.Experiment,data_segment_key=None):
    # return
    PARAM=exper.get_param()
    SAVE_DIR=exper.get_save_dir()

    # train_trans=get_train_transform()

    assert(data_segment_key in [None, "base","novel"] )
    # train_sample_process=ClassficationSampleProcss(PARAM.data.img_size,
    #                                                PARAM.data.means,PARAM.data.stds,train_trans)
    class_names=PARAM.data.class_names
    target_size=(PARAM.data.input_shape,PARAM.data.input_shape)
    train_sample_process=ClipPreprocess(class_names,augment=True,target_size=target_size)
    if PARAM.data.roi_size_list is not None :
        dataset = MultiviewRoiData(root=PARAM.data.root,split="train",
                                         roi_size_list=PARAM.data.roi_size_list,
                                         mv_method=PARAM.data.mv_method,
                                          config_dir = PARAM.data.config_name,
        sample_process=train_sample_process,min_size=PARAM.data.min_size)
        dataset.data_name= PARAM.data.data_name
    else:
        assert False 
        dataset = RoiFabricData(root=PARAM.data.root,split="train",config_dir = PARAM.data.config_name,
        sample_process=train_sample_process,min_size=PARAM.data.min_size)
        data_name= PARAM.data.data_name
    if data_segment_key=="base":
        dataset.filter_by_class_names(PARAM.data.base_class_names)
    if data_segment_key=="novel":
        dataset.filter_by_class_names(PARAM.data.novel_class_names)



    # train_dataset=DatasetWrapper(dataset,{"x":"image","y":"label_num","bboxes":"box_list"})

    # from .part_data import PartDatasetTool
    # few_shot_data,_ =PartDatasetTool(dataset=dataset,class_key="label_num").get_suport_query_data(5)
    train_dataset=DatasetWrapper(dataset,{"x":"image","y":"label_num","bboxes":"box_list","masks":"masks"})

    # # print(1111111111)
    sampler=None
    # weights=train_dataset.dataset.get_sample_wights()
    # # # print(len(weights))
    # sampler=WeightedRandomSampler(weights=weights,num_samples=len(train_dataset),
    #                           replacement=True)

    train_dataloader_param=PARAM.train_dataloader.clone()
    train_dataloader_param.sampler=sampler
    train_dataloader_param.shuffle=False
    train_dataloader_param.dataset=train_dataset
    train_dataloader=DataLoader(**train_dataloader_param,collate_fn= dataset.get_collate_fn() )
    return train_dataset,train_dataloader
from functools import partial
create_train_data_novel= partial(create_train_data,data_segment_key="novel")
create_train_data_base= partial(create_train_data,data_segment_key="base")

def create_valid_data(exper:FM.Experiment,data_segment_key=None):
    PARAM=exper.get_param()
    SAVE_DIR=exper.get_save_dir()
    assert(data_segment_key in [None, "base","novel"] )
    
    # train_sample_process=ClassficationSampleProcss(PARAM.data.img_size,
    #                                                PARAM.data.means,PARAM.data.stds,train_trans)
    class_names=PARAM.data.class_names
    target_size=(PARAM.data.input_shape,PARAM.data.input_shape)
    train_sample_process=ClipPreprocess(class_names,target_size=target_size,augment=False)
    if PARAM.data.roi_size_list is not None :
        dataset = MultiviewRoiData(root=PARAM.data.root,split="valid",
                                         roi_size_list=PARAM.data.roi_size_list,
                                         mv_method=PARAM.data.mv_method,
                                         config_dir = PARAM.data.config_name,
        sample_process=train_sample_process,min_size=PARAM.data.min_size)
        data_name= PARAM.data.data_name
    else:
        assert False
        dataset = RoiFabricData(root=PARAM.data.root,split="valid",config_dir = PARAM.data.config_name,
        sample_process=train_sample_process,min_size=PARAM.data.min_size)
        data_name= PARAM.data.data_name

    if data_segment_key=="base":
        dataset.filter_by_class_names(PARAM.data.base_class_names)
    if data_segment_key=="novel":
        dataset.filter_by_class_names(PARAM.data.novel_class_names)

    valid_data=DatasetWrapper(dataset,{"x":"image","y":"label_num","bboxes":"box_list","masks":"masks"})

    return valid_data

create_valid_data_novel= partial(create_valid_data,data_segment_key="novel")
create_valid_data_base= partial(create_valid_data,data_segment_key="base")



# def round_to_nearest_integer(tensor,min_val,max_val):
#     rounded_tensor = torch.round(tensor)
#     clamped_tensor = torch.clamp(rounded_tensor, min=min_val, max=max_val)
#     return clamped_tensor

import numpy as np
from lyus.Frame.trainer import KeyTrainer
from lyus.Frame.optim import OptimizerKit
from lyus.Frame import  Mapper



def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if hasattr(p,"grad") and p.grad is not None:
            p.grad.data = p.grad.data.float()


# def train_step(self: KeyTrainer):
#     # get the next batch of input data from the data loader
#     input_batch = next(self.train_data_loader)
#     # move the input data to the device (GPU or CPU)
#     input_batch = Mapper(lambda x: x.to(self.device))(input_batch)

#     # pass the input data to the model and get the output results
#     self.model.zero_grad()
#     result_batch = self.model(input_batch)
#     # get the losses from the output results
#     losses = result_batch["losses"]

    
#     # update the optimizer with the losses and check if it is the end of the epoch
#     self.optimizer.step(sum(losses.values()).detach().cpu().numpy(),
#                         epoch_end=(self.epo_step == self.num_steps_per_epoch))


#     # create a dictionary to store the variables for this step
#     step_variables = {"epo": self.epo, "lr": self.optimizer.get_lr()}
#     # create a function to move tensors from device to cpu
#     to_cpu = Mapper(lambda x: x.detach().cpu())
#     # add the input batch variables to the step variables dictionary
#     for key, val in input_batch.items():
#         step_variables[key] = to_cpu(val)
#     # add the output result variables to the step variables dictionary
#     for key, val in result_batch.items():
#         step_variables[key] = to_cpu(val)
#     # assign the step variables dictionary to the attribute of the class
#     self.step_variables = step_variables


from lyus.Frame.metrics import  CommonMetricHook,classificationMetric,OrdinalClassMetric,BinaryClassMetric,AucMetric
from lyus.Frame.metrics import Accucary



from lyus.Frame.hooks import  HookBase
class AdjustModelModeHook(HookBase):

    def __init__(self, finetune_epoch=30):
        self.finetune_epoch = finetune_epoch

    def before_epoch(self):
        self.trainer.model.set_mode("ssl")
        # if self.trainer.epo <= self.finetune_epoch:
        #     self.trainer.model.set_mode("finetune")
        # else:
        #     self.trainer.model.set_mode("train")




from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from lyus.Frame.hooks import HookBase




from lyus.data import CsvLabelData
from lyus.Frame import DatasetWrapper
from lyus.Frame.checkpoint import CheckPointHook, CheckPoint


class CommonMetricHookPlus(CommonMetricHook):

    
    def after_train(self):

        
        model= self.trainer.model
        model.set_mode("eval")
        metrics= self.eval_run(model)
        # for k ,val in metrics.items():
        #     self.trainer.epo_variables[self.metric_name+k]=val
        # epo_name="_".join(["epo",str(self.trainer.step)])
        epo_name="last epo "
        for key ,val in metrics.items():
            key = "_".join([self.data_name, key])
            self.trainer.epo_variables[key]=val
            # Experiment().add_scalar(key, val, self.trainer.step)
            FM.Experiment().info("_".join([epo_name,str(key) ,str(val),]))



        # self.trainer.model.set_img_prototype(5, self.trainer.train_data_loader)

        # metrics= self.eval_run(model)
        # # for k ,val in metrics.items():
        # #     self.trainer.epo_variables[self.metric_name+k]=val
        # # epo_name="_".join(["epo",str(self.trainer.step)])
        # epo_name="after learn image proto, last "
        # for key ,val in metrics.items():
        #     key = "_".join([self.data_name, key])
        #     self.trainer.epo_variables[key]=val
        #     # Experiment().add_scalar(key, val, self.trainer.step)
        #     FM.Experiment().info("_".join([epo_name,str(key) ,str(val),]))

        # self.trainer.model.set_img_text_proto()
        # metrics= self.eval_run(model)
        # # for k ,val in metrics.items():
        # #     self.trainer.epo_variables[self.metric_name+k]=val
        # # epo_name="_".join(["epo",str(self.trainer.step)])
        # epo_name="after learn image proto, last1  "
        # for key ,val in metrics.items():
        #     key = "_".join([self.data_name, key])
        #     self.trainer.epo_variables[key]=val
        #     # Experiment().add_scalar(key, val, self.trainer.step)
        #     FM.Experiment().info("_".join([epo_name,str(key) ,str(val),]))



def build_hooks(exper:FM.Experiment,valid_data):
    PARAM = exper.get_param()
    hooks=[]
    # hooks.append(InputVisualHook(3, PARAM.data.means,PARAM.data.stds))
    hooks.append(AdjustModelModeHook(**PARAM.hook.AdjustModelModeHook))


    hooks.append(CheckPointHook(**PARAM.hook.CheckPointHook))


    num_classes=PARAM.data.num_classes
    hooks.append(CommonMetricHookPlus(valid_data,classificationMetric(num_classes),
                                  "test",pred_name="predicts",period=1,label_name="y"))

    return hooks



def get_mddel_param(exper:FM.Experiment,model):
    PARAM = exper.get_param()

    # layer2_params = model.backbone.parameters()

    # # 获取其余层的参数
    # base_params = filter(lambda p: id(p) not in list(map(id, layer2_params)), model.parameters())
    # parameters = [
    #     {'params': base_params},  # 默认学习率
    #     {'params': layer2_params, 'lr': PARAM.optim.Adam.lr * 0.001}  # 特定层的学习率是其余层的10倍
    # ]
    parameters=model.parameters()
    return parameters

 