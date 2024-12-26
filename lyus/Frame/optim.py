import numpy as np

import lyus.Frame as FM
import torch.optim as optim
from torch.optim import lr_scheduler


__all__=[
    "OptimizerKit",
    "HyperParam",
    ]





class OptimizerKit(object):

    def __init__(self, parameters,**params):
        self.params = FM.Param(**params)
        # print(self.params)
        self.__batches_loss = []
        self.__epoches_loss = []
        self.__epoches_end = []
        self.__optimizer = None
        self.__lrReducer = None
        self.__earlyStopper = None
        self.__warmUp = None
        self.__add_parameters(parameters)

    def __add_parameters(self, parameters):

        self.__parameters = parameters
        # self.__parameters = filter(lambda p: p.requires_grad, parameters)

        # init optimizer
        assert hasattr(self.params, "optimizer_name") and hasattr(self.params, self.params.optimizer_name)
        OPTIMIZER=eval("optim."+self.params.optimizer_name)
        self.__optimizer = OPTIMIZER(params=self.__parameters,**self.params[self.params.optimizer_name])
    
        # init lr reducer
        if hasattr(self.params, 'lrReducer_name') and self.params.lrReducer_name != None:
            assert hasattr(self.params, self.params.lrReducer_name)
            LR_SCHEDULER= eval("lr_scheduler." + self.params.lrReducer_name)
            self.__lrReducer= LR_SCHEDULER(optimizer=self.__optimizer,**self.params[self.params.lrReducer_name])

        else:
            self.__LrReducer = None

        # init early stopper
        if hasattr(self.params, "with_earlyStopper") and self.params.with_earlyStopper:
            assert hasattr(self.params, "EarlyStopper")
            self.__earlyStopper = EarlyStopper(**self.params.EarlyStopper)
        else:
            self.__earlyStopper = None

        if hasattr(self.params, "with_warmUp") and self.params.with_warmUp:
            assert hasattr(self.params, "WarmUp")
            warmup_param=FM.Param(**self.params.WarmUp)
            self.__warmUp = WarmUp(optimizer=self.__optimizer,**self.params.WarmUp)
        else:
            self.__warmUp = None

    def step(self, batch_loss, epoch_end=False):
        self.__optimizer.step()
        
        # culate loss and iteration number
        self.__batches_loss.append(batch_loss)
        if epoch_end:
            epoch_begin = self.__epoches_end[-1] if len(self.__epoches_end) > 0 else 0
            epoch_end = len(self.__batches_loss)
            epoch_loss = sum(self.__batches_loss[epoch_begin:epoch_end])
            self.__epoches_end.append(epoch_end)
            self.__epoches_loss.append(epoch_loss)
        num_batches = len(self.__batches_loss)
        num_epoches = len(self.__epoches_loss)

        # warmup
        if self.__warmUp != None:
            self.__warmUp.step(num_batches)
        # forward


        # reduce lr
        if epoch_end and self.__lrReducer != None:
            # self.__lrReducer.step(self.__epoches_loss[-1], num_epoches)
            self.__lrReducer.step(num_epoches)

        # wether early stop
        if epoch_end and self.__earlyStopper != None:
            self.__earlyStopper.step(self.__epoches_loss[-1])

    # user should stop training  if get  a  "true"
    def whetherEarlyStop(self):
        return self.__earlyStopper.early_stop

    def get_lr(self):

        return self.__optimizer.state_dict()['param_groups'][0]['lr']


class HyperParam(object):
    def __init__(self, initVal, finalVal=-1., beginStep=-1, endStep=-1):
        self.initVal = initVal
        self.finalVal = finalVal
        self.beginStep = beginStep
        self.endStep = endStep
        assert self.beginStep<=self.endStep
    def __call__(self, step, **kwargs):
        val = self.initVal
        if self.beginStep == self.endStep:
            return val
        step = step + (self.beginStep - step) * (self.beginStep > step) + (self.endStep - step) * (self.endStep < step)
        val += self.riseCosin((step - self.beginStep) / (self.endStep - self.beginStep)) * (self.finalVal - self.initVal)
        return val

    def riseCosin(self,x):
        return (np.cos((x + 1) * np.pi) + 1) / 2


class WarmUp(object):

    def __init__(self,optimizer,min_lr,num_steps,method='line',verbose=True):
        self.optimizer=optimizer
        self.min_lr=min_lr
        self.num_steps=num_steps
        self.verbose=verbose
        self.max_lr=[float(param_group['lr'])  for i, param_group in enumerate(self.optimizer.param_groups)]

    def step(self,idx_step):
        if idx_step>self.num_steps:
            return
        for i, param_group in enumerate(self.optimizer.param_groups):
            # new_lr = self.min_lr + (self.max_lr[i] - self.min_lr) * self.get_alpha(idx_step)
            new_lr=HyperParam(self.min_lr,self.max_lr[i],0,self.num_steps)(idx_step)
            param_group['lr'] = new_lr
            if self.verbose:
                print('Step {:5d}: WarmUp learning rate'
                      ' of group {} to {:.4e}.'.format(idx_step, i, new_lr))

    def get_alpha(self,idx_step):
        return 1



class EarlyStopper:
    """Early stops the training if  loss doesn't improve after a given patience."""
    def __init__(self,thred_loss ,patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.thred_loss = -thred_loss
        self.early_stop = False
        self.delta = delta

    def step(self, loss):
        score = -loss
        if score > self.thred_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True








if __name__ == '__main__':


    optim_param=FM.Param(

        optimizer_name="SGD",
        lrReducer_name="ReduceLROnPlateau",
        with_earlyStopper=True,
        with_warmUp=True,
    )
    optim_param.SGD=FM.Param()
    optim_param.Adam=FM.Param()
    optim_param.WarmUp=FM.Param()
    optim_param.ReduceLROnPlateau=FM.Param()

    parameters={}
    optimer=OptimizerKit(parameters,optim_param)


    




