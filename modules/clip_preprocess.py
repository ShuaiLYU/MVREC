

import os,sys


from .utils import Bbox,RoiGenerator,resize_img_and_bbox,resize_img



import numpy as np
from  PIL import  Image
import  re
import  torch
import random
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,ColorJitter
def _convert_image_to_rgb(image):
    return image.convert("RGB")

class ClipPreprocess(object):


    def __init__(self,class_names,target_size=None,augment=False,device=None,**kwargs):
        self.target_size=target_size #(512,512)
        self.augment=augment
        self.class_names=class_names
        self.visual_mode=kwargs.get("visual_mode",False)
        self.preprocess=self.get_clip_process()

        
        self.mask_transform = Compose([
            ToTensor(), 
            # Resize((224, 224)),
            Normalize(0.5, 0.26)
        ])

        # self.device=device
        self.device=device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
    def get_clip_process(self):
        
        if self.visual_mode:
            return None
        def _transform(n_px):
            if not self.augment:
                return Compose([
                    # Resize(n_px, interpolation=BICUBIC),
                    # CenterCrop(n_px),
                    _convert_image_to_rgb,
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])
            else:
                return Compose([
                    # Resize(n_px, interpolation=BICUBIC),
                    # CenterCrop(n_px),
                    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
                    _convert_image_to_rgb,
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])

        return _transform(None)
    

    
    
    def get_label_info(self, label):

        index = self.class_names.index(label)
        text="a photo of {} defect on the image".format(label)
        return index,text
    
    
    def __call__(self, sample):

        img = sample["roi_img"]
        bbox = sample["bboxes"][0].get()
        mask=sample["masks"][0]
        if self.target_size:
            img, bbox = resize_img_and_bbox(img, bbox, self.target_size)
            mask= resize_img(mask, self.target_size)
        # print(self.target_size,img.size,111)
        label=sample["bboxes"][0].label
        bbox_list=[bbox]
        if self.preprocess:
            bbox_list = torch.tensor(bbox_list).to(self.device) 
            img = self.preprocess(img).to(self.device)    
            mask =  self.mask_transform(mask) .to(self.device).float()      
        index,text=self.get_label_info(label)
        return {
            "image":img,
            "box_list":bbox_list,
            "masks":mask,
            "label":label,
            "text": text,
            "src_bbox":sample["src_bbox"],
            "img_path":sample["img_path"],
            "label_num":index
        }
    # def get(self):



def get_train_transform():
    
    param=FM.Param()
    param.ColorJitter = FM.Param(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)
    param.RandomHorizontalFlip = FM.Param(p=0.5)
    param.DiscreteRotateTransform = FM.Param(angles=[0,90,180,270])
    param.AutoAugment=None
    param.RandomCenterCrop=FM.Param(scale=(0.6,1.2),p=0.7)

    trans=[]
    if param.RandomHorizontalFlip is not None:
        trans.append(transforms.RandomHorizontalFlip(**param.RandomHorizontalFlip))
    if param.ColorJitter is not None:
        trans.append(transforms.ColorJitter(**param.ColorJitter))

    # if param.AutoAugment is None:
    #     trans.append(autoaugment.AutoAugment(**param.AutoAugment))
    if param.RandomCenterCrop is not None:
        trans.append(my_trans.RandomCenterCrop(**param.RandomCenterCrop))
    if param.DiscreteRotateTransform is not None:
        trans.append(my_trans.DiscreteRotateTransform(**param.DiscreteRotateTransform))

    return trans

class ClassficationSampleProcss(object):
    
    
    def __init__(self,img_size,mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),train_transforms=list(),withlabel=True):

        tensor_transforms=[transforms.Resize(img_size),transforms.ToTensor(),transforms.Normalize(mean,std)]

        self.trans=my_trans.Compose(train_transforms+tensor_transforms)
        self.withlabel=withlabel

    # def get(self,img_name):
    #     img_bboxes = self.sample_by_img[img_name]
    #     img_path=img_bboxes[0]["imagePath"]
    #     img_id=img_bboxes[0]["img_id"]
    #     img= Image.open(img_path)

    #     #take the whole image as roi
    #     roi_bbox=Bbox().from_dict({"x1":0,"y1":0,"x2":img.width,"y2":img.height,"imageWidth":img.width,
    #                                "imageHeight":img.height})
    #     # get the roi-corped sample
    #     roi_sample= { "roi_bbox":roi_bbox,"roi_img":img,"bboxes":img_bboxes,"img_id":img_id,"img_name":img_name,
    #                  "imgSz":(roi_bbox.Width(),roi_bbox.Height()) }

    #     # sample process  callback function (add transformers  here )
    #     if self.sample_process is not None:
    #         roi_sample=self.sample_process(roi_sample)
    #     return roi_sample


    def __call__(self,sample):


        img=self.trans(sample["roi_img"])
        bboxes=sample["bboxes"]
        img_name=sample["img_name"]
        label=bboxes[0].label
        sample={"img_name":img_name,"bboxes":bboxes,"img":img}
        return  sample