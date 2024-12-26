from FabricDataLib.fabric_data import RoiFabricData
from FabricDataLib.utils import BaseSampleProcess,RoiGenerator,resize_img_and_bbox,resize_img,super_crop

from PIL import Image

import numpy as np 
import torch 

import cv2
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt



# class MvParam:
#     def __init__(self, roi_size_list, x_step_list=[0], y_step_list=[0], angles=[0]):
#         self.roi_size_list = roi_size_list
#         self.x_step_list = x_step_list
#         self.y_step_list = y_step_list
#         self.angles = angles
#         self.param_list, self.param_name_list = self._generate_params()

#     def _generate_params(self):
#         params = []
#         for size in self.roi_size_list:
#             for j in self.y_step_list:
#                 for i in self.x_step_list:
#                     for angle in self.angles:
#                         params.append([size, i, j, angle])

#         named_params = self._name_params(params)
        
#         param_list = [param for name, param in named_params]
#         param_name_list = [name for name, param in named_params]

#         return param_list, param_name_list

#     def _name_params(self, params):
#         named_params = []
#         for param in params:
#             size, x_step, y_step, angle = param
#             name = f"size_{size}_x_{x_step}_y_{y_step}_angle_{angle}"
#             named_params.append((name, param))
#         return named_params

#     def find_param_index(self, sizes=None, x_steps=None, y_steps=None, angles=None):
#         indices = []
#         for index, name in enumerate(self.param_name_list):
#             match = True
#             if sizes is not None and not any(f"size_{size}_" in name for size in sizes):
#                 match = False
#             if x_steps is not None and not any(f"_x_{x_step}_" in name for x_step in x_steps):
#                 match = False
#             if y_steps is not None and not any(f"_y_{y_step}_" in name for y_step in y_steps):
#                 match = False
#             if angles is not None and not any(f"_angle_{angle}" in name for angle in angles):
#                 match = False
#             if match:
#                 indices.append(index)
        
#         return indices
class MvParam:
    def __init__(self, roi_size_list, x_step_list=[0], y_step_list=[0], angles=[0], flips=[0]):
        self.roi_size_list = roi_size_list
        self.x_step_list = x_step_list
        self.y_step_list = y_step_list
        self.angles = angles
        self.flips = flips
        self.param_list, self.param_name_list = self._generate_params()

    def _generate_params(self):
        params = []
        for size in self.roi_size_list:
            for j in self.y_step_list:
                for i in self.x_step_list:
                    for angle in self.angles:
                        for flip in self.flips:
                            params.append([size, i, j, angle, flip])

        named_params = self._name_params(params)
        
        param_list = [param for name, param in named_params]
        param_name_list = [name for name, param in named_params]

        return param_list, param_name_list

    def _name_params(self, params):
        named_params = []
        for param in params:
            size, x_step, y_step, angle, flip = param
            name = f"size_{size}_x_{x_step}_y_{y_step}_angle_{angle}_flip_{flip}"
            named_params.append((name, param))
        return named_params

    def find_param_index(self, sizes=None, x_steps=None, y_steps=None, angles=None, flips=None):
        indices = []
        for index, name in enumerate(self.param_name_list):
            match = True
            if sizes is not None and not any(f"size_{size}_" in name for size in sizes):
                match = False
            if x_steps is not None and not any(f"_x_{x_step}_" in name for x_step in x_steps):
                match = False
            if y_steps is not None and not any(f"_y_{y_step}_" in name for y_step in y_steps):
                match = False
            if angles is not None and not any(f"_angle_{angle}_" in name for angle in angles):
                match = False
            if flips is not None and not any(f"_flip_{flip}" in name for flip in flips):
                match = False
            if match:
                indices.append(index)
        
        return indices

    @staticmethod
    def rotate_img(img, angle, roi):
        if angle == 0:
            return img
        img=np.array(img)
        x1, y1, x2, y2 = roi
        roi_center_x = (x1 + x2) // 2
        roi_center_y = (y1 + y2) // 2
        roi_center = (roi_center_x, roi_center_y)
        
        # Get the dimensions of the image
        height, width = img.shape[:2]
        
        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(roi_center, angle, 1.0)
        
        # Perform the rotation
        rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
        
        return Image.fromarray(rotated_img)

    @staticmethod
    def flip_img(img, flip):
        img=np.array(img)
        if flip == 0:
            # No flip
            # return img
            pass
        elif flip == 1:
            # Horizontal flip
            img= cv2.flip(img, 1)
        elif flip == 2:
            # Vertical flip
            img=  cv2.flip(img, 0)
        elif flip == 3:
            # Flip along left diagonal (transpose)
            img=  cv2.transpose(cv2.flip(img, -1))
        elif flip == 4:
            # Flip along right diagonal (transpose)
            img=  cv2.transpose(img)
        else:
            raise ValueError("Invalid flip value. It should be 0, 1, 2, 3, or 4.")
        return Image.fromarray(img)

class MultiScaleRoiGenerator(RoiGenerator):

    def __init__(self,roi_size_list,x_step_list,y_step_list ,angles,flips,target_size=None):
        self.roi_size_list=roi_size_list
        self.target_size=target_size
        # self.mvParam= MvParam(roi_size_list,x_step_list=[-1,0,1], y_step_list=[-1,0,1])

        self.mvParam= MvParam(roi_size_list,x_step_list=x_step_list, y_step_list=y_step_list,
         angles = angles,flips=flips)
    def __call__(self, img,bbox,mask):
        roi_list,img_roi_list, roi_bbox_list,roi_mask_list=  self.gen_one(img,
        bbox, self.mvParam.param_list, mask)
        if self.target_size:
            for i, (img_roi,roi_bbox,roi_mask) in enumerate(zip(img_roi_list,roi_bbox_list,roi_mask_list)):
                img_roi_list[i], roi_bbox_list[i]=resize_img_and_bbox(img_roi,roi_bbox,
                                                                      (self.target_size,self.target_size))
                roi_mask_list[i]=resize_img(roi_mask,(self.target_size,self.target_size))
        return roi_list,img_roi_list, roi_bbox_list,roi_mask_list
    
    @staticmethod 
    def gen_roi_size_list(img, bbox,shape_factors=[2,3,4]):
        bbox= [ int(x) for  x in bbox]
        imgSize=img.size
        imgW,imgH=imgSize
        centerX,centerY,bboxw,bboxh=MultiScaleRoiGenerator.bbox2rect(bbox)
        bbox_min_size=min(bboxw,bboxh)
        img_min_size=min(imgW,imgH)
        roi_size_base= min(bbox_min_size,img_min_size//max(*shape_factors) )
        roi_size_list= [roi_size_base*m  for m in  shape_factors]  
        return roi_size_list



    @staticmethod 
    def gen_one(img, bbox,param_list,mask=None):

        centerX,centerY,bboxw,bboxh=MultiScaleRoiGenerator.bbox2rect(bbox)
        # print(roi_size_list)
        # assert False
        roi_list,img_roi_list, roi_bbox_list,mask_roi_list= [],[],[],[]

        for param in param_list:
            size,i,j,angle,flip =param

            if float(size).is_integer():
                size_x,size_y=size,size
                
            else:
                size_x= abs(bbox[2] - bbox[0]) *float(size)
                size_y=  abs( bbox[3] - bbox[1])*float(size)
                
            center_x=centerX+ i* size_x//3
            center_y=centerY+ j* size_y//3
            roi = [int(center_x-size_x//2), int(center_y-size_y//2),
                int(center_x-size_x//2+size_x), int(center_y-size_y//2+size_y) ] 
  
            roi_w=roi[2]-roi[0]
            roi_h=roi[3]-roi[1]
            roi_bbox = (max(bbox[0] - roi[0],0), max(bbox[1] - roi[1],0),
                        min(bbox[2] - roi[0],roi_w),min(bbox[3] - roi[1],roi_h))

            #% 这里增加过程， 对 img 和mask 进行旋转
            img=MvParam.flip_img(img,flip)

            mask=MvParam.flip_img(mask,flip)
            img=MvParam.rotate_img(img,angle,roi)
            mask=MvParam.rotate_img(mask,angle,roi)

            img_roi=super_crop(img,roi)
            if mask is not None :
                mask_roi=super_crop(mask,roi)
                mask_roi_list.append(mask_roi)
            else:
                mask_roi_list.append(None)
            roi_list.append(roi)
            img_roi_list.append(img_roi)
            roi_bbox_list.append(roi_bbox)
        return roi_list,img_roi_list, roi_bbox_list,mask_roi_list
 
        
import time
from functools import wraps

def log_execution_time(filename="execution_times.txt"):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Start timing
            start_time = time.perf_counter()
            
            # Call the actual function
            result = func(self, *args, **kwargs)
            
            # Stop timing
            elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
            
            # Log to the specified file
            with open(filename, "a") as file:
                class_name = self.__class__.__name__
                method_name = func.__name__
                file.write(f"{class_name}.{method_name} executed in {elapsed_time:.3f} ms\n")
            
            return result
        return wrapper
    return decorator



class MultiScaleRoiSampleProcess(BaseSampleProcess):

    def __init__(self,roi_size_list,target_size=None):
        self.target_size=target_size

                # angles=[0, 45, 90, 135, 180, 225, 270, 315]
        angles=[0]# [0, 45, 135, 225, 315]
        x_step_list=[0]# [-1,0,1]
        y_step_list=[0] # [-1,0,1]
        flips=[0] # [0,1,2,3,4]
        self.roi_gen=MultiScaleRoiGenerator(roi_size_list=roi_size_list,
                                            x_step_list=x_step_list,
                                            y_step_list=y_step_list,
                                            angles=angles,
                                            flips=flips,
                                            target_size=target_size)

    def __call__(self,src_bbox):
        # print(bbox.kwargs)
        # recall the whole image and all defects on them
        img_name=src_bbox.kwargs["imageName"]
        img_id=src_bbox.kwargs["img_id"]
        img_path = src_bbox.kwargs["imagePath"]

        src_img=Image.open(img_path)

        points =src_bbox.kwargs.get("points",None)
        if  points is not None:
            src_mask=self.point2mask(points,src_bbox.imgsz.w,src_bbox.imgsz.h)
        else:
            src_mask=None 
        roi_samples=[]
        # @log_execution_time("time_log.txt")
        def run(self):
            return self.roi_gen(src_img,src_bbox.get(),src_mask)
        roi_bboxs,roi_imgs,bbox_in_rois,roi_mask_list=run(self)
        for roi_bbox,roi_img,bbox_in_roi,roi_mask in zip(roi_bboxs,roi_imgs,bbox_in_rois,roi_mask_list):
            bbox=src_bbox.clone()
            bbox.x1=bbox_in_roi[0]
            bbox.y1=bbox_in_roi[1]
            bbox.x2=bbox_in_roi[2]
            bbox.y2=bbox_in_roi[3]
            bbox.imgsz.w=roi_img.width
            bbox.imgsz.h=roi_img.height
            bbox.mask=roi_mask

 
                # Extract the ROI coordinates
    
            roi_sample= {"roi_bbox":roi_bbox, "roi_img":roi_img,"bboxes":[bbox], "imgSz":(roi_img.width,roi_img.height),"src_img":src_img,"src_bbox":src_bbox.get(),"masks":[bbox.mask],
                         "img_path":img_path,"img_id":img_id,"img_name":img_name}
            roi_samples.append(roi_sample)
        return roi_samples

    def point2mask(self,points, img_w, img_h):

        # Create a blank mask
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        # print(points)
        if isinstance(points, str):
            points = eval(points)
        # Convert points to a numpy array of shape (n_points, 1, 2)
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        
        # Fill the polygon on the mask
        cv2.fillPoly(mask, [pts], color=255)
            
        return mask



def custom_collate_fn(batch):
    """
    Custom collate function to merge batch samples where each sample is a dictionary
    containing lists of tensors under various keys (e.g., 'x', 'y', etc.). Tensors for each key
    are concatenated along the batch dimension.
    
    Args:
    batch (list): A list of data samples where each sample is a dict with keys mapping to lists of tensors.
    
    Returns:
    dict: A dictionary with each key having concatenated tensors along the batch dimension.
    """
    # 初始化结果字典，这里将每个键的张量列表初始化为空列表
    collated_dict = {}

    stack_keys= ["x", "bboxes","masks"]
    # 遍历batch中的每一个样本
    for sample in batch:
        # 再遍历样本中的每个键
        for key, value in sample.items():
            if key not in collated_dict: collated_dict[key] = []
            if key in stack_keys:
        
                # 将当前样本的张量列表扩展到相应的键中
                collated_dict[key].append(torch.stack(value,dim=0))
            else:
                    collated_dict[key].append(value[0])

    # 对每个键的列表进行批次维度上的连接
    for key, value in collated_dict.items():
        if torch.is_tensor(value[0]):
            collated_dict[key] = torch.stack(value, dim=0)
        else:
            collated_dict[key] = torch.tensor(value)

    return collated_dict


# class RoiSampleProcess(MultiScaleRoiSampleProcess):

#     def __init__(self,roi_size_list,target_size=None):
#         self.target_size=target_size

#         roi_size_list=[roi_size_list[0]]
#         angles=[0]# [0, 45, 135, 225, 315]
#         x_step_list=[0]# [-1,0,1]
#         y_step_list=[0] # [-1,0,1]
#         flips=[0,1,2,3,4]
#         self.roi_gen=MultiScaleRoiGenerator(roi_size_list=roi_size_list,
#                                             x_step_list=x_step_list,
#                                             y_step_list=y_step_list,
#                                             angles=angles,
#                                             flips=flips,
#                                             target_size=target_size)



class RoiSampleProcess(MultiScaleRoiSampleProcess):

    def __init__(self,roi_size_list,target_size=None):
        self.target_size=target_size

        roi_size_list=[roi_size_list[0]]
        angles=[0]# [0, 45, 135, 225, 315]
        x_step_list=[0]# [-1,0,1]
        y_step_list=[0] # [-1,0,1]
        flips=[0]
        self.roi_gen=MultiScaleRoiGenerator(roi_size_list=roi_size_list,
                                            x_step_list=x_step_list,
                                            y_step_list=y_step_list,
                                            angles=angles,
                                            flips=flips,
                                            target_size=target_size)

class MultiRotateRoiSampleProcess(MultiScaleRoiSampleProcess):

    def __init__(self,roi_size_list,target_size=None):
        self.target_size=target_size

        roi_size_list=[roi_size_list[0]]
        angles=[0,90,180,270]# [0, 45, 135, 225, 315]
        x_step_list=[0]# [-1,0,1]
        y_step_list=[0] # [-1,0,1]
        flips=[0] #[0,1,2,3,4]
        self.roi_gen=MultiScaleRoiGenerator(roi_size_list=roi_size_list,
                                            x_step_list=x_step_list,
                                            y_step_list=y_step_list,
                                            angles=angles,
                                            flips=flips,
                                            target_size=target_size)



class MultiFlipRoiSampleProcess(MultiScaleRoiSampleProcess):

    def __init__(self,roi_size_list,target_size=None):
        self.target_size=target_size

        roi_size_list=[roi_size_list[0]]
        angles=[0] # [0,90,180,270]# [0, 45, 135, 225, 315]
        x_step_list=[0]# [-1,0,1]
        y_step_list=[0] # [-1,0,1]
        flips=[0,1,2,3,4]
        self.roi_gen=MultiScaleRoiGenerator(roi_size_list=roi_size_list,
                                            x_step_list=x_step_list,
                                            y_step_list=y_step_list,
                                            angles=angles,
                                            flips=flips,
                                            target_size=target_size)

class MultiOffsetRoiSampleProcess(MultiScaleRoiSampleProcess):

    def __init__(self,roi_size_list,target_size=None):
        self.target_size=target_size
        roi_size_list=[roi_size_list[0]]
                # angles=[0, 45, 90, 135, 180, 225, 270, 315]
        angles=[0]# [0, 45, 135, 225, 315]
        x_step_list=[-1,0,1]
        y_step_list= [-1,0,1]
        flips=[0] # [0,1,2,3,4]
        self.roi_gen=MultiScaleRoiGenerator(roi_size_list=roi_size_list,
                                            x_step_list=x_step_list,
                                            y_step_list=y_step_list,
                                            angles=angles,
                                            flips=flips,
                                            target_size=target_size)
        

class MultiScaleOffsetRoiSampleProcess(MultiScaleRoiSampleProcess):

    def __init__(self,roi_size_list,target_size=None):
        self.target_size=target_size

                # angles=[0, 45, 90, 135, 180, 225, 270, 315]
        angles=[0]# [0, 45, 135, 225, 315]
        x_step_list=[-1,0,1]
        y_step_list=[-1,0,1]
        flips=[0] # [0,1,2,3,4]
        self.roi_gen=MultiScaleRoiGenerator(roi_size_list=roi_size_list,
                                            x_step_list=x_step_list,
                                            y_step_list=y_step_list,
                                            angles=angles,
                                            flips=flips,
                                            target_size=target_size)

class MultiScaleOffsetBboxSampleProcess(MultiScaleRoiSampleProcess):

    def __init__(self,roi_size_list,target_size=None):
        self.target_size=target_size
        roi_size_list=[0.999,1.4999,1.999]
                # angles=[0, 45, 90, 135, 180, 225, 270, 315]
        angles=[0]# [0, 45, 135, 225, 315]
        x_step_list=[-1,0,1]
        y_step_list=[-1,0,1]
        flips=[0] # [0,1,2,3,4]
        self.roi_gen=MultiScaleRoiGenerator(roi_size_list=roi_size_list,
                                            x_step_list=x_step_list,
                                            y_step_list=y_step_list,
                                            angles=angles,
                                            flips=flips,
                                            target_size=target_size)
        
class MultiScaleFlipRoiSampleProcess(MultiScaleRoiSampleProcess):

    def __init__(self,roi_size_list,target_size=None):
        self.target_size=target_size

                # angles=[0, 45, 90, 135, 180, 225, 270, 315]
        angles=[0]# [0, 45, 135, 225, 315]
        x_step_list=[0] #[-1,0,1]
        y_step_list=[0]# [-1,0,1]
        flips= [0,1,2,3,4]
        self.roi_gen=MultiScaleRoiGenerator(roi_size_list=roi_size_list,
                                            x_step_list=x_step_list,
                                            y_step_list=y_step_list,
                                            angles=angles,
                                            flips=flips,
                                            target_size=target_size)

class MultiScaleRotateRoiSampleProcess(MultiScaleRoiSampleProcess):

    def __init__(self,roi_size_list,target_size=None):
        self.target_size=target_size

        # angles=[0, 45, 90, 135, 180, 225, 270, 315]
        angles=[0]# [0, 45, 135, 225, 315]
        angles=[0,90,180,270]
        x_step_list=[0] #[-1,0,1]
        y_step_list=[0]# [-1,0,1]
        flips=[0]# # [0,1,2,3,4]
        self.roi_gen=MultiScaleRoiGenerator(roi_size_list=roi_size_list,
                                            x_step_list=x_step_list,
                                            y_step_list=y_step_list,
                                            angles=angles,
                                            flips=flips,
                                            target_size=target_size)

class MultiOffsetRotateRoiSampleProcess(MultiScaleRoiSampleProcess):

    def __init__(self,roi_size_list,target_size=None):
        self.target_size=target_size
        roi_size_list=[roi_size_list[0]]
        # angles=[0, 45, 90, 135, 180, 225, 270, 315]
        # angles=[0]# [0, 45, 135, 225, 315]
        angles=[0,90,180,270]
        x_step_list=[-1,0,1]
        y_step_list=[-1,0,1]
        flips=[0]# # [0,1,2,3,4]
        self.roi_gen=MultiScaleRoiGenerator(roi_size_list=roi_size_list,
                                            x_step_list=x_step_list,
                                            y_step_list=y_step_list,
                                            angles=angles,
                                            flips=flips,
                                            target_size=target_size)


class MultiviewRoiFabricData(RoiFabricData):
    def __init__(self,root, split: str,roi_size_list:list,mv_method="mso", config_dir=None, sample_process=None,target_size=512, **kwargs):
        self.target_size = target_size
        super().__init__( root, split, config_dir, sample_process, **kwargs)

        mv_method_map= {
            "none": "RoiSampleProcess",
            "ms":"MultiScaleRoiSampleProcess",
            "mo":"MultiOffsetRoiSampleProcess",
            "mr":"MultiRotateRoiSampleProcess",
            "mf":"MultiFlipRoiSampleProcess",
            "mso":"MultiScaleOffsetRoiSampleProcess",
            "msf":"MultiScaleFlipRoiSampleProcess",
            "msr":"MultiScaleRotateRoiSampleProcess",
            "mor":"MultiOffsetRotateRoiSampleProcess",
            "mso_bbox": "MultiScaleOffsetBboxSampleProcess"  # crop roi according to the defect size
            }
        assert mv_method in mv_method_map.keys(),mv_method
        self.mv_method=mv_method
        self.roi_process=eval(mv_method_map[mv_method])(roi_size_list=roi_size_list,target_size=self.target_size)
        self.sam_process=sample_process
        self.roi_size_list=roi_size_list
        # self.multiView=3
    def __len__(self):
        return len(self.sample_by_instance)

    def __getitem__(self, index):
        import time
        bbox_src= self.sample_by_instance[index]
        # Start timing
        start_time = time.perf_counter()
        bboxes=self.roi_process(bbox_src) # self.mv_method
        # Stop timing
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        # Print the method name and time taken in milliseconds
        # print(f"{self.mv_method} executed in {elapsed_time:.3f} ms")

        for ind,bbox in enumerate(bboxes):
            if self.sam_process:
                bboxes[ind]=self.sam_process(bbox)

        def listdict2dictlist(bboxes):
            bboxes = {key: list(values) for key, values in 
                                zip(bboxes[0].keys(), zip(*[bbox.values() for bbox in bboxes]))}
            return bboxes
  

        return listdict2dictlist(bboxes)
    
    # def get_collate_fn(self):

    #     def custom_collate_fn(batch):
    #         """
    #         Custom collate function to merge batch samples where each sample is a dictionary
    #         containing lists of tensors under various keys (e.g., 'x', 'y', etc.). Tensors for each key
    #         are concatenated along the batch dimension.
            
    #         Args:
    #         batch (list): A list of data samples where each sample is a dict with keys mapping to lists of tensors.
            
    #         Returns:
    #         dict: A dictionary with each key having concatenated tensors along the batch dimension.
    #         """
    #         # 初始化结果字典，这里将每个键的张量列表初始化为空列表
    #         collated_dict = {}

    #         # 遍历batch中的每一个样本
    #         for sample in batch:
    #             # 再遍历样本中的每个键
    #             for key, value in sample.items():
    #                 if key not in collated_dict: collated_dict[key] = []
    #                 # 将当前样本的张量列表扩展到相应的键中
    #                 collated_dict[key].extend(value)

    #         # 对每个键的列表进行批次维度上的连接
    #         for key, value in collated_dict.items():
    #             if torch.is_tensor(value[0]):
    #                 collated_dict[key] = torch.stack(value, dim=0)
    #             else:
    #                 collated_dict[key] = torch.tensor(value)


    #         return collated_dict
        
    #     return custom_collate_fn
    
    def get_collate_fn(self):

        self.custom_collate_fn=custom_collate_fn
        return self.custom_collate_fn