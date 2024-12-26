
from .base import *


class CheckpointHook(HookBase):
    def after_train(self):
        # self.trainer是在钩子注册的时候定义的
        epoch = self.trainer.epo
        # if epoch % 3 == 0:
        print(epoch)
            

            
from .utils import InverseNormalize,ToNumpy
class InputVisualHook(HookBase):
    
    def __init__(self,num_epoch,mean,std):
        
        self.num_epoch=num_epoch
        self.inverse=InverseNormalize(mean,std)
        self.toNp=ToNumpy(transpose=True)
    def after_epoch(self):
        epoch=self.trainer.epo
        if epoch==1 or epoch%self.num_epoch==0:
            img_batch=self.trainer.step_variables["x"]
            img_batch=self.inverse(img_batch)
            img_batch=self.toNp(img_batch)
            file_name_batch=[ str(i)+".jpg" for i in range(len(img_batch)) ]
            self.trainer.saver.visualize([img_batch],"epo_{}".format(epoch),file_name_batch)
