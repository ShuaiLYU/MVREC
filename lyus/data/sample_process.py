
import numpy as np
import torch 
import torch.nn as nn
import os
import random
import pandas as pd

from torchvision import transforms
from PIL import Image
from lyus.torch_utils.transforms import Compose,RandomCenterCrop,DiscreteRotateTransform

class ClassficationSampleProcss(object):
    
    
    def __init__(self,img_size,mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),train_transforms=list()):
        assert config in ["train","valid"]

        tensor_transforms=[transforms.Resize(img_size),transforms.ToTensor(),transforms.Normalize(mean,std)]

        self.trans=Compose(train_transforms+tensor_transforms)



                 
    def __call__(sample):

        def read_img(img_path):
            img_path=img_path.replace("\\","/")
            image= Image.open(img_path).convert("RGB")
            return image
                 
        img=read_img(sample["filepath"])
        img=self.trans(img)
        return {"img":img,"filepath":sample["filepath"],"label":label}   
    
    def get_train_transform(self,param):
        trans=[]
        if param.RandomHorizontalFlip is not None:
            trans.append(transforms.RandomHorizontalFlip(**param.RandomHorizontalFlip))
        if param.ColorJitter is not None:
            trans.append(transforms.ColorJitter(**param.ColorJitter))

        # if param.AutoAugment is None:
        #     trans.append(autoaugment.AutoAugment(**param.AutoAugment))
        if param.RandomCenterCrop is not None:
            trans.append(RandomCenterCrop(**param.RandomCenterCrop))
        if param.DiscreteRotateTransform is not None:
            trans.append(DiscreteRotateTransform(**param.DiscreteRotateTransform))

        return trans

    
    


