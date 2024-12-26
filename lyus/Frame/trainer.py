from lyus.Frame import Experiment
import  torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .base import TrainerBase,HookBase,VariableBase

from .optim import OptimizerKit
from .saver import  Saver
from .utils import Mapper



class commonTrainer(TrainerBase):
    
    
    
    def load(self,
        model: nn.Module,
        optimizer: OptimizerKit,
        train_data_loader: DataLoader,
        saver: Saver,
        loss_dict:dict,
        device,**kwargs):
        self.model=model
        self.model.to(device)
        self.optimizer=optimizer
        self.saver=saver
        self.device=device
        self.param=kwargs.get("param",None)
        self.num_steps_per_epoch=len(train_data_loader)
        def get_data():
            while True:
                for out in train_data_loader:
                    yield out

        self.train_data_loader=get_data()
        self.loss_dict=loss_dict
        self.step_variables={}
        self.epo_variables= {}
    
    def train_step(self):
        x_batch,y_batch=next(self.train_data_loader)
        
        # run_map can handle tensor, list(tensor) ,dict(tensor)
        x_batch=Mapper(lambda x: x.to(self.device))(x_batch)
        y_batch=Mapper(lambda x: x.to(self.device))(y_batch)
        fx_batch = self.model(x_batch)
        # update_mask_batch=mask_tensor.detach().cpu().numpy()
        ###### cul loss
        loss_dict={}
        if isinstance(self.loss_dict, nn.Module):
            loss_dict=self.loss_dict(fx_batch,y_batch)
            if not isinstance(loss_dict,dict): loss_dict={"loss":loss_dict}
        elif isinstance(self.loss_dict,dict) :
            loss_dict= { name:loss_func(fx_batch,y_batch) for name,loss_func in self.loss_dict.items()}
        elif isinstance(self.loss_dict,list):
            loss_dict = {idx: loss_func(fx_batch, y_batch) for idx, loss_func in enumerate(self.loss_dict)}
        self.model.zero_grad()
        batch_loss_sum = sum(loss_dict.values())
        batch_loss_sum.backward()
        self.optimizer.step(batch_loss_sum.detach().cpu().numpy(),epoch_end=(self.epo_step==self.num_steps_per_epoch))
        step_variables={}
        # step_variables["x_batch"]=x_batch
        to_cpu=Mapper(lambda x: x.detach().cpu())
        step_variables["x_batch"]=to_cpu(x_batch)
        step_variables["y_batch"]=to_cpu(y_batch)
        step_variables["fx_batch"]=to_cpu(fx_batch)
        step_variables["loss_dict"]=to_cpu(loss_dict)
        step_variables["batch_loss_sum"]=batch_loss_sum.item()
        step_variables["epo"]=self.epo
        step_variables["lr"]= self.optimizer.get_lr()
        self.step_variables=step_variables
        
        
        
    def before_train(self):
        self.saver.info(self.model)
        if self.param is not None: self.saver.info(self.param)
        TrainerBase.before_train(self)
    def after_step(self):
        TrainerBase.after_step(self)
        for key, val in  self.step_variables.items():
            if key in ["lr","batch_loss_sum","epo"]:
                self.saver.add_scalar(key,val,self.step)
        for key,val in   self.step_variables["loss_dict"].items():
            self.saver.add_scalar(key,val,self.step)
        
    def before_epoch(self):
        self.epo_variables = {}
        TrainerBase.before_epoch(self)

    def after_epoch(self):
        TrainerBase.after_epoch(self)
        # print(self.epo_variables)
        for key, val in  self.epo_variables.items():
            self.saver.add_scalar(key,val,self.step)

        


class KeyTrainer(TrainerBase):
    def load(self,
             model: nn.Module,
             optimizer: OptimizerKit,
             train_data_loader: DataLoader,
             device, **kwargs):

        self.model = model
        self.model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.param = kwargs.get("param", None)
        self.num_steps_per_epoch=len(train_data_loader)
        def get_data():
            while True:
                for out in train_data_loader:
                    yield out

        self.train_data_loader = get_data()
        self.step_variables = {}
        self.epo_variables = {}

    def train_step(self):
        # get the next batch of input data from the data loader
        input_batch = next(self.train_data_loader)
        # move the input data to the device (GPU or CPU)
        input_batch = Mapper(lambda x: x.to(self.device))(input_batch)

        # pass the input data to the model and get the output results
        self.model.zero_grad()
        result_batch = self.model(input_batch)
        # get the losses from the output results
        losses=result_batch["losses"]
        # update the optimizer with the losses and check if it is the end of the epoch
        self.optimizer.step(sum(losses.values()).detach().cpu().numpy(),
                            epoch_end=(self.epo_step == self.num_steps_per_epoch))
        # create a dictionary to store the variables for this step
        step_variables = {"epo":self.epo,"lr": self.optimizer.get_lr()}
        # create a function to move tensors from device to cpu
        to_cpu = Mapper(lambda x: x.detach().cpu())
        # add the input batch variables to the step variables dictionary
        for key,val in input_batch.items():
            step_variables[key]= to_cpu(val)
        # add the output result variables to the step variables dictionary
        for key,val in result_batch.items():
            step_variables[key]= to_cpu(val)
        # assign the step variables dictionary to the attribute of the class
        self.step_variables = step_variables

    def after_step(self):
        # call the after_step method of the base class
        TrainerBase.after_step(self)
        # import the partial and DictExecutor classes
        from functools import  partial
        from lyus.Frame.utils import  DictExecutor
        # create a partial function that adds a scalar to the saver with the global step
        add_scalar_func=partial(Experiment().add_scalar,global_step=self.step)
        # create a DictExecutor object that applies the add_scalar_func to a dictionary
        add_scalar_exeacutor=DictExecutor(add_scalar_func)
        # apply the add_scalar_exeacutor to the step variables dictionary
        add_scalar_exeacutor(self.step_variables)
        
        
    def before_train(self):
        Experiment().info(self.model)
        TrainerBase.before_train(self)

        
    def before_epoch(self):
        self.epo_variables = {}
        TrainerBase.before_epoch(self)
        # self.model.set_mode("train")

    def after_epoch(self):
        TrainerBase.after_epoch(self)
        # print(self.epo_variables)
        for key, val in  self.epo_variables.items():
            Experiment().add_scalar(key,val,self.step)

        if len(self.epo_variables)>0:
            self.epo_variables["epo"]=self.epo
            Experiment().add_train_var(self.epo_variables)



