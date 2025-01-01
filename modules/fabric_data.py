import os.path
from PIL import  Image
import  enum
import numpy as np 
import cv2
class SPLIT(enum.Enum):
    TRAIN=0
    VALID=1

split_map={"train":SPLIT.TRAIN,"valid":SPLIT.VALID}

from .utils import CsvLabelData,Bbox,RoiGenerator
from collections import  OrderedDict
def _convert_image_to_rgb(image):
    return image.convert("RGB")
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
class FabricData(object):
    def __init__(self, root,split:str, config_dir="",sample_process=None, **kwargs):


        self.root =root
        assert  split in ["train","valid"]
        self.split = split_map[split]
        self.sample_process= sample_process
        self.config_dir=config_dir
        # self._check_data_config(self.configs)
        self.load()


    # def _check_data_config(self,configs:dict) -> bool:
    #     all_parts= Folder(self.root).find_child_folders(False)
    #     assert all_parts is not None
    #     for part_name,config_name in configs.items():
    #         assert os.path.exists(os.path.join(self.root,part_name)),"data subset: {} is not existing! "
    #         all_folders = Folder(os.path.join(self.root, part_name)).find_child_folders(False)
    #         assert "images" in all_folders, "images folder is not existing in data subset : {}".format(part_name)
    #         assert config_name in all_folders, "config: {} is not existing in data subset : {}".format(config_name,part_name)
    #         train_config= os.path.join(self.root,part_name,config_name,"train.csv")
    #         valid_config = os.path.join(self.root, part_name, config_name, "valid.csv")
    #         assert os.path.exists(train_config), "train.csv config: {} is not existing in data subset : {}".format(config_name,part_name)
    #         assert os.path.exists(valid_config), "valid.csv config: {} is not existing in data subset : {}".format(config_name, part_name)

    def load(self):

        # print(self.root,self.config_name,11111111111111)
        def load_sample_process(data:dict):
            # print(self.root,data["part"],data["img_rel_path"])
            data["imagePath"]= os.path.join(self.root,data["part"],data["img_rel_path"])
            data["img_id"]= data["img_id"]
            data["id"]= data["id"]
            data["part"]= data["part"]
            bbox=Bbox().from_dict(data)
            return bbox
        
        # load all defect instance from csv
        sample_by_instance=[]       
        if self.split==SPLIT.TRAIN:
            csv_config= os.path.join(self.config_dir,"train.csv")
            sample_by_instance += CsvLabelData(csv_config,sample_process=load_sample_process).get()
            # print(sample_by_instance[0])
        elif self.split==SPLIT.VALID:
            csv_config = os.path.join(self.config_dir, "valid.csv")
            sample_by_instance += CsvLabelData(csv_config,load_sample_process).get()

        # group sample by img_name, push all instances within an image together
        sample_by_img=OrderedDict()
        for sam in sample_by_instance:
            img_name= sam["imageName"]
            if img_name not in sample_by_img.keys(): sample_by_img[img_name]=[]
            sample_by_img[img_name].append(sam)
        self.sample_by_instance=sample_by_instance
        self.sample_by_img=sample_by_img
        self.img_names= list(sample_by_img.keys())

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name=self.img_names[index]
        return self.get(img_name)


    def imgname2id(self,img_name):
        img_bboxes = self.sample_by_img[img_name]
        return img_bboxes[0]["img_id"]

    
    def get(self,img_name):
        img_bboxes = self.sample_by_img[img_name]
        img_path=img_bboxes[0]["imagePath"]
        img_id=img_bboxes[0]["img_id"]
        img= Image.open(img_path)

        #take the whole image as roi
        roi_bbox=Bbox().from_dict({"x1":0,"y1":0,"x2":img.width,"y2":img.height,"imageWidth":img.width,
                                   "imageHeight":img.height})
        # get the roi-corped sample
        roi_sample= { "roi_bbox":roi_bbox,"roi_img":img,"bboxes":img_bboxes,"img_id":img_id,"img_name":img_name,
                     "imgSz":(roi_bbox.Width(),roi_bbox.Height()) }

        # sample process  callback function (add transformers  here )
        if self.sample_process is not None:
            roi_sample=self.sample_process(roi_sample)
        return roi_sample
    

# class RoiFabricData(FabricData):
#     def __init__(self,root, split: str, config_dir=None, sample_process=None,min_size=512, **kwargs):
#         self.min_size = min_size
#         super().__init__( root, split, config_dir, sample_process, **kwargs)
#         self.random_size= False if split =="valid" else True
#         # self.random_size= False
#         self.random_offset= False if split =="valid" else True
    
#     def __len__(self):
#         return len(self.sample_by_instance)

#     def __getitem__(self, index):
#         bbox= self.sample_by_instance[index]

#         # recall the whole image and all defects on them
#         img_name=bbox.kwargs["imageName"]
#         img_id=bbox.kwargs["img_id"]
#         img_path = bbox.kwargs["imagePath"]
#         img=Image.open(img_path)
        
#         # take of current bbox as anchor and create a roi bbox
#         # roi_bbox=bbox.pad_and_random_move(self.min_size)
#         # # read image and crop roi
#         # roi_img= img.crop(roi_bbox.get())
#         # new_bboxs=[bbox.crop_by_roi(roi_bbox)]
#         # crop all the boxes within the roi region
#         # new_bboxs=[]
#         # for bbox in  img_ins:
#         #     if roi_bbox.IoU(bbox)>0:
#         #         new_bboxs.append(bbox.crop_by_roi(roi_bbox))

#         src_bbox=bbox.get()
#         roi_bbox,roi_img,bbox_in_roi=RoiGenerator.gen_one(img,bbox.get(),self.min_size,max_size=self.min_size*10,
#                                                           random_size=self.random_size,random_offset=self.random_offset)
#         bbox=bbox.clone()
#         bbox.x1=bbox_in_roi[0]
#         bbox.y1=bbox_in_roi[1]
#         bbox.x2=bbox_in_roi[2]
#         bbox.y2=bbox_in_roi[3]
#         bbox.imgsz.w=roi_img.width
#         bbox.imgsz.h=roi_img.height
#         # get the roi-coped sample
#         roi_sample= {"roi_bbox":roi_bbox, "roi_img":roi_img,"bboxes":[bbox], "imgSz":(roi_img.width,roi_img.height),
#                      "src_bbox":src_bbox,"img_path":img_path,
#                      "img_id":img_id,"img_name":img_name}
#         roi_sample["img"]=img
#         # sample process  callback function (add transformers  here )
#         if self.sample_process is not None:
#             roi_sample=self.sample_process(roi_sample)
#         return roi_sample

    def get_collate_fn(self):
        return None
    

    

from .utils import BaseSampleProcess

class RoiSampleProcess(BaseSampleProcess):

    def __init__(self,roi_size,random_size=False,random_offset=False):
        self.roi_size=roi_size
        self.random_size=random_size
        self.random_offset=random_offset
        
    def __call__(self,src_bbox):
        # print(bbox.kwargs)
        # recall the whole image and all defects on them
        img_name=src_bbox.kwargs["imageName"]
        img_id=src_bbox.kwargs["img_id"]
        img_path = src_bbox.kwargs["imagePath"]
        src_img=Image.open(img_path)
        bbox=src_bbox.clone()
        roi_img=src_img
        use_roi=True
        roi_bbox=None
        if  use_roi:
            roi_bbox,roi_img,bbox_in_roi=RoiGenerator.gen_one(src_img,src_bbox.get(),
                                                            min_size=self.roi_size,
                                                            max_size=self.roi_size*10,
                                                            random_size=self.random_size,
                                                            random_offset=self.random_offset)
     
            bbox.x1=bbox_in_roi[0]
            bbox.y1=bbox_in_roi[1]
            bbox.x2=bbox_in_roi[2]
            bbox.y2=bbox_in_roi[3]
            bbox.imgsz.w=roi_img.width
            bbox.imgsz.h=roi_img.height
        # get the roi-coped sample
        roi_sample= {"roi_bbox":roi_bbox, "roi_img":roi_img,"bboxes":[bbox], "imgSz":(roi_img.width,roi_img.height),
                     "src_img":src_img,"src_bbox":src_bbox.get(),"img_path":img_path,"img_id":img_id,"img_name":img_name}

        return roi_sample


from tqdm import tqdm


class RoiFabricData(FabricData):
    def __init__(self,root, split: str, config_dir=None, sample_process=None,min_size=512, **kwargs):
        self.min_size = min_size
        super().__init__( root, split, config_dir, sample_process, **kwargs)
        self.random_size= False if split =="valid" else True
        self.random_size= False
        self.random_offset= False if split =="valid" else True
        self.roi_process=RoiSampleProcess(self.min_size,self.random_size,self.random_offset)
        self.sam_process=sample_process
    def __len__(self):
        return len(self.sample_by_instance)

    def __getitem__(self, index):
        bbox= self.sample_by_instance[index]
        bbox=self.roi_process(bbox)
        if self.sam_process:
            bbox=self.sam_process(bbox)
        return bbox
    
    def filter_by_class_names(self, class_names):
        print("Sample count before filtering:")
        self.print_class_counts()
        filtered_samples = [sam for sam in self.sample_by_instance if sam.label in class_names]
        self.sample_by_instance = filtered_samples
        print("Sample count after filtering:")
        self.print_class_counts()

    def print_class_counts(self):
        from collections import Counter
        all_value = [sam.label for sam in self.sample_by_instance]
        class_counts = dict(Counter(all_value))
        for label, count in class_counts.items():
            print(f"Class {label}: {count} samples")


    def get_sample_wights(self):
        from collections import Counter
        all_value = [sam.label for sam in self.sample_by_instance ]
        # print(self.cls_dict)
        value_num = dict(Counter(all_value))
        num_total = len(self)
        weights = []
        # print(len(self))
        for val in all_value:
            w = num_total / value_num[val]
            weights.append(w)
        return weights

    def get_class_indices(self):
        class_indices = {}
        for i, sample in tqdm(enumerate(self.sample_by_instance)):
            label = sample.label
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
        return class_indices







