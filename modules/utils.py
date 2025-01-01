import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from PIL import Image
def random_bbox(image, min_size=20, max_size=300):
    import numpy as np
    width, height = image.size
    # max
    w = np.random.randint(min_size, min(max_size,image.width) )
    h = np.random.randint(min_size, min(max_size,image.height))
    x1 = np.random.randint(0, width - w)
    y1 = np.random.randint(0, height - h)
    x2 = x1 + w
    y2 = y1 + h
    return (x1, y1, x2, y2)



def truncate_bbox(bbox, length):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    if width > length:
        x2 = x1 + length

    if height > length:
        y2 = y1 + length

    return [x1, y1, x2, y2]


# def super_crop(img, roi):
#     # 支持 坐标超出图像边界，
#     x1, y1, x2, y2 = roi
#     width, height = img.size
#     new_img = Image.new('RGB', (x2 - x1, y2 - y1))
#     left = max(0, x1)
#     top = max(0, y1)
#     right = min(width, x2)
#     bottom = min(height, y2)
#     new_img.paste(img.crop((left, top, right, bottom)), (left - x1, top - y1))
#     return new_img

from PIL import Image
import numpy as np

def super_crop(img, roi):
    if isinstance(img, np.ndarray):
        is_array = True
        img = Image.fromarray(img)
    else:
        is_array = False

    x1, y1, x2, y2 = roi
    width, height = img.size
    channels = img.mode

    new_img = Image.new(channels, (x2 - x1, y2 - y1))

    left = max(0, x1)
    top = max(0, y1)
    right = min(width, x2)
    bottom = min(height, y2)

    crop_img = img.crop((left, top, right, bottom))

    # Paste the cropped region onto the new image
    new_img.paste(crop_img, (left - x1, top - y1))

    if is_array:
        return np.array(new_img)
    else:
        return new_img



import cv2
def resize_img(img, imgSz):
    """
    Resize the input image to the given size.
    
    Args:
    - img: Input image, either a PIL.Image or a NumPy array.
    - imgSz: Tuple (width, height) for the new size.
    
    Returns:
    - Resized image, same type as the input.
    """
    if isinstance(img, Image.Image):
        # Input is a PIL.Image
        resized_img = img.resize(imgSz)
        return resized_img
    elif isinstance(img, np.ndarray):
        # Input is a NumPy array
        resized_img = cv2.resize(img, imgSz)
        return resized_img
    else:
        raise TypeError("Unsupported image type. Input should be a PIL.Image or a NumPy array.")

def resize_img_and_bbox(img: Image.Image, bbox, imgSz):
    # 获取图像的原始尺寸
    original_width, original_height = img.size

    # 计算图像的缩放因子
    width_scale = imgSz[0] / original_width
    height_scale = imgSz[1] / original_height

    # 调整图像的尺寸
    resized_img = img.resize(imgSz)

    # 更新边界框的坐标
    resized_bbox = [
        int(bbox[0] * width_scale),  # x1
        int(bbox[1] * height_scale),  # y1
        int(bbox[2] * width_scale),  # x2
        int(bbox[3] * height_scale)  # y2
    ]

    return resized_img, resized_bbox


class ImgRoi(object):
    def __init__(self, img_array):
        self.img_array = img_array
        # self.bbox=bbox

    def crop(self, roi):
        # 支持 坐标超出图像边界，
        img = self.img_array
        roi = [int(value) for value in roi]
        x1, y1, x2, y2 = roi
        width, height = img.size
        new_img = Image.new('RGB', (x2 - x1, y2 - y1))
        left = max(0, x1)
        top = max(0, y1)
        right = min(width, x2)
        bottom = min(height, y2)
        new_img.paste(img.crop((left, top, right, bottom)), (left - x1, top - y1))
        return new_img

    def crop_by_center(self, center, w, h):
        x1 = center[0] - w // 2
        y1 = center[1] - h // 2
        bbox = [x1, y1, x1 + w, y1 + h]
        # print(w,h,bbox)d
        return self.crop(bbox)
class RoiGenerator(object):

    def __init__(self,min_size=512,max_size=None,target_size=None,random_offset=False,random_size=False,fixed_center=False):
        self.min_size=min_size
        if max_size is None:
            max_size=self.min_size*10
        self.max_size=max_size
        self.random_offset=random_offset
        self.random_size=random_size
        self.target_size=target_size
        self.fixed_center=fixed_center
    def __call__(self, img,bbox):
        roi,img_roi, roi_bbox=  self.gen_one(img,bbox,self.min_size,self.max_size
                                             ,self.random_offset,self.random_size,self.fixed_center)
        if self.target_size: img_roi, roi_bbox=resize_img_and_bbox(img_roi,roi_bbox,(self.target_size,self.target_size))
        return roi,img_roi,roi_bbox

    @staticmethod
    def bbox2rect(bbox): # [x1,y1,x2,y2]  ->  [centerX,centerY, W, H]
        centerX = (bbox[0] + bbox[2]) // 2
        centerY = (bbox[1] + bbox[3]) // 2
        bboxW= abs(bbox[2] - bbox[0])
        bboxH= abs(bbox[3] - bbox[1])
        return [centerX,centerY,bboxW,bboxH]
    

    @staticmethod
    def getLocRange(size, inRange,outRange):
        # inRange[0] >= x - size /2   -> 
        # outRange[0] <= x - size /2 
        #  inRange[1] <= x - size /2 +size
        #  outRange[1] >= x - size /2 +size
        # print(size,inRange,outRange)
        inRange=(max(inRange[0],outRange[0]),min(inRange[1],outRange[1]))
        low= max(outRange[0]+size//2, inRange[1]+size//2-size)
        up= min ( inRange[0]+ size//2 , outRange[1]+size//2-size)
        assert low <=up,(low,up)
        # center= max(low,center)
        # center=min(center,up)
        return (int(low),int(up))
    @staticmethod 
    def gen_one(img, bbox,min_size,max_size,random_offset=False,random_size=True,fix_center=False):
        bbox= [ int(x) for  x in bbox]
        def getSizeRange(imgSize,bboxSize):
            imgW,imgH=imgSize
            # print(imgW,imgH,max_size)
            max_size1=min(min(imgW,imgH),max_size)

            bboxw,bboxh=bboxSize
            min_size1 = max(max(bboxw,bboxh),min_size)

            sizeRange=[min(min_size1, max_size1), max_size1]
            return sizeRange
    
        imgSize=img.size
        imgW,imgH=imgSize
        centerX,centerY,bboxw,bboxh=RoiGenerator.bbox2rect(bbox)
        
        ################ 1.  cal  size Range and size
        min_size, max_size=getSizeRange(imgSize,(bboxw,bboxh))
        size=min_size
        if random_size: size=random.randint(min_size,max_size)
        # print(min_size, max_size)
        ################ 2. refine bbox size by the size param
        lineX= (bbox[0],bbox[2])
        if bboxw>size: lineX= ( centerX-size//2,centerX-size//2+size )
        lineY= (bbox[1],bbox[3])
        if bboxh>size: lineY= ( centerY-size//2,centerY-size//2+size )
        ################ 3. cal the roi center range  and  roi center

        # print(random_offset)
        if not fix_center:
            rangeX = RoiGenerator.getLocRange(size, lineX, (0, imgW))
            rangeY = RoiGenerator.getLocRange(size, lineY, (0, imgH))
            if not random_offset:
                centerX= min(max(centerX,rangeX[0]),rangeX[1])
                centerY= min(max(centerY,rangeY[0]),rangeY[1])
            else:
                centerX=random.randint(*rangeX)
                centerY=random.randint(*rangeY)
        
        ################ 4. cal ROI 
        #  centerX-size//2 >=0   centerX+ (size-size//2) <= imgW
        roi = [centerX-size//2, centerY-size//2,centerX-size//2+size, centerY-size//2+size ] 
        # self.roi=roi
        roi_w=roi[2]-roi[0]
        roi_h=roi[3]-roi[1]
        ################ 5. get roi_bbox and  img_roi
        roi_bbox = (max(bbox[0] - roi[0],0), max(bbox[1] - roi[1],0),
                    min(bbox[2] - roi[0],roi_w),min(bbox[3] - roi[1],roi_h))

        from PIL import Image

        img_roi=super_crop(img,roi)

        return roi,img_roi, roi_bbox



    

import os ,shutil
import random
from functools import  partial

import copy
import  numpy as np


def my_truncat(x, min_val, max_val):
    return min(max(x, min_val), max_val)
class LabelConfigTool():

    @staticmethod
    def get_divide_indexes(data:list,train_ratio, valid_ratio, shuffle=True):

        ratio_dict = {"train": train_ratio, "valid": valid_ratio, "test": 1 - train_ratio - valid_ratio}
        train_offset = int(np.floor(len(data) * ratio_dict["train"]))
        val_offset = int(np.floor(len(data) * (ratio_dict["train"] + ratio_dict["valid"])))
        indexes=["train"]*train_offset+["valid"]*(val_offset-train_offset)
        if (len(data)-val_offset)>0:
            indexes += ["test"]*(len(data)-val_offset)
        if shuffle:
            random.shuffle(indexes)
        return indexes

    @staticmethod
    def sample_list_to_csv(samples,csvname=None):
        assert isinstance(samples,list) and len(samples)>0
        for sam in samples:
            assert  isinstance(sam,dict)
        # print(samples[0])
        keys=list(samples[0].keys())
        data=[]
        for col in range(len(samples)):
            sample=samples[col]
            item=[  sample[keys[i]] for i in range(len(keys) )]
            data.append(item)

        data=pd.DataFrame(data,columns=keys)
        if csvname is not None:
            data.to_csv(csvname,index=False)
        return data


class Folder(object):

    def __init__(self, root):
        self.root = root

    def get_child_folders(self):
        try :
            return next(os.walk(self.root))[1]
        except:
            return None

    def exists(self, filename):
        return os.path.exists(os.path.join(self.root, filename))

    def find_files_by_suffix(self, suffixes, recursion=False):

        def condition_func(filename, suffix):
            return filename.endswith(suffix)

        if not isinstance(suffixes, (list, tuple)):
            suffixes = [suffixes]
        res = []
        for suffix in suffixes:
            condition = partial(condition_func, suffix=suffix)
            res += self.list_folder(self.root, True, condition, recursion)
        return res

    def find_child_folders(self, return_path=True,condition=None):

        dirs = [{"root": root, "dirs": dirs, "files": files} for root, dirs, files in os.walk(self.root)][0]["dirs"]
        if condition is not None:
            dirs = [d for d in dirs if condition(d)]
        if return_path:
            dirs = [os.path.join(self.root, d) for d in dirs]
        return dirs

    def find_file(self, filename):
        for root, _, _ in os.walk(self.root):
            filepath = os.path.join(root, filename)
            if os.path.exists(filepath):
                return filepath
        return None

    def copy_to(self, new_dir):
        shutil.copytree(self.root, new_dir)

    """

    os.remove(path)   #删除文件
    os.removedirs(path)   #删除空文件夹

    os.rmdir(path)    #删除空文件夹

    shutil.rmtree(path)    #递归删除文件夹，即：删除非空文件夹
    """

    def _deep_delete_file(self, func):
        for root, dirs, files in os.walk(self.root):
            for name in files:
                if (func(name)):
                    os.remove(os.path.join(root, name))
                    print("delect : {}  in  {} ".format(name, root))

    def _deep_delete_folder(self, folder_name):
        for root, dirs, files in os.walk(self.root):
            for dir in dirs:
                if (dir == folder_name):
                    cur_path = os.path.join(root, dir)
                    shutil.rmtree(cur_path)
                    print("remove : {}".format(cur_path))



    @staticmethod
    def list_folder(root, use_absPath=True, func=None, recursion=False):
        """
        :param root:  文件夹根目录
        :param func:  定义一个函数，过滤文件
        :param use_absPath:  是否返回绝对路径， false ：返回相对于root的路径
        :return:
        """
        root = os.path.abspath(root)
        if os.path.exists(root):
            print("遍历文件夹【{}】......".format(root))
        else:
            raise Exception("{} is not existing!".format(root))
        files = []
        # 遍历根目录,
        for cul_dir, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                path = os.path.join(cul_dir, fname)  # .replace('\\', '/')
                if func is not None and not func(path):
                    continue
                if use_absPath:
                    files.append(path)
                else:
                    files.append(os.path.relpath(path, root))
            if not recursion:
                break
        print("    find {} file under {}".format(len(files), root))
        return files


import pandas as pd


class CsvLabelData(object):

    def __init__(self, csv_path, sample_process=None):
        self.samples = self.csv_to_sample_list(csv_path)
        self.sample_process = sample_process

    def csv_to_sample_list(self, csvname):
        data = pd.read_csv(csvname)
        data.fillna('', inplace=True)
        return data.apply(pd.Series.to_dict, axis=1).to_list()

    def get(self):
        return [ sam for sam in self]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.sample_process is None:
            return self.samples[idx]
        else:
            return self.sample_process(self.samples[idx])

    def get_wightSampler_wight(self, key):
        from collections import Counter
        all_value = [sam[key] for sam in self.samples]
        # print(self.cls_dict)
        value_num = dict(Counter(all_value))
        num_total = len(self.samples)
        weights = []
        # print(len(self))
        for data in self.samples:
            w = num_total / value_num[data[key]]
            weights.append(w)
        return weights



class ImgSZ(object):
    def __init__(self, img=None):
        if img is not None:
            img = np.array(img)
            self.w, self.h = img.shape[1], img.shape[0]

    def init_with_size(self, img_size: tuple):
        self.w, self.h = img_size[0], img_size[1]
        return self

    def get(self):
        return self.w, self.h






class Bbox(object):

    def _init(self, x1, y1, x2, y2, label, imgsz: ImgSZ = None, **kwargs):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label
        self.imgsz = imgsz
        self.kwargs = kwargs
        self.points=[]
        return self

    @staticmethod
    def to_llm_rect_shape(label, bbox):
        def bbox_to_points(bbox):
            x1, y1, x2, y2 = bbox
            points = [[x1, y1], [x2, y2]]
            return points

        shape_keys = ["label", "points", "shape_type", "flags", "group_id"]
        shape = dict(
            label=label,
            points=bbox_to_points(bbox),
            shape_type="rectangle",
            flags={},
            group_id=None,
            other_data={}
        )
        return shape
 
    def __getitem__(self, attr):
        
        if attr in self.__dict__.keys():
            return self.__dict__[attr]
        elif attr in self.kwargs.keys():
            return self.kwargs[attr]
        else:
            return None
    
    def to_llm_shape(self):
        return self.to_llm_rect_shape(self.label, [self.x1, self.y1, self.x2, self.y2])

    def from_llm_shape(self, shape,imgsz=ImgSZ(),extra_shape_keys=[],**kwargs):
        self.imgsz=imgsz
        # assert (shape['shape_type'] == "rectangle" or shape['shape_type'] == "proposal")
        self.shape = copy.deepcopy(shape)

        def points_to_bbox(points):
            """
            将 LabelMe 的 points 列表转换为 bbox（bounding box）。
            
            参数:
                points (list): LabelMe 格式的 points 列表，每个点是一个 [x, y] 列表。
            
            返回:
                tuple: 表示 bbox 的 (x_min, y_min, x_max, y_max) 元组。
            """
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            return x_min, y_min, x_max, y_max

        self.x1, self.y1, self.x2, self.y2 = points_to_bbox(shape["points"])
        self.points= shape["points"] 
        self.label = shape["label"]
        self.rank= shape.get("rank",-1)
        self.kwargs=kwargs
        for key in extra_shape_keys:
            self.kwargs[key]=shape.get(key,None)
        return self

    def from_dict(self,data:dict):
        self.x1=data["x1"]
        self.x2 = data["x2"]
        self.y1 = data["y1"]
        self.y2 = data["y2"]
        self.label=data.get("label",-1)
        self.rank = data.get("rank",-1)
        self.kwargs=data
        self.imgsz=ImgSZ().init_with_size((data["imageWidth"],data["imageHeight"]))
        return self


    def to_dict(self)->dict:
        data= copy.deepcopy(self.kwargs)
        # print(self.kwargs)
        data["x1"]=self.x1
        data["x2"]=self.x2
        data["y1"]=self.y1
        data["y2"]=self.y2
        data["label"]=self.label
        data["rank"] = self.rank
        data["points"]=self.points
        return data



    #     def to_llm_shape(self):
    #         assert( hasattr(self,"shape"))
    #         def bbox_to_points(bbox):
    #             return [[bbox[0],bbox[1]],[bbox[2],bbox[3]]]
    #         self.shape["points"]=bbox_to_points([self.x1,self.y1,self.x2,self.y2])
    #         return copy.deepcopy(self.shape)

    def get(self):
        return [self.x1, self.y1, self.x2, self.y2]

    def Width(self):
        return self.x2-self.x1
    def Height(self):
        return self.y2-self.y1
    def area(self):
        return max(0., self.x2 - self.x1) * max(0., self.y2 - self.y1)

    def overlap(self, bbox):
        return self._overlap(self.get(), bbox.get())

    @staticmethod
    def _overlap(box1, box2):
        '''
        计算两个矩形框的交并比
        :param box1: list,第一个矩形框的左上角和右下角坐标
        :param box2: list,第二个矩形框的左上角和右下角坐标
        :return: 两个矩形框的交并比iou
        '''
        x1 = max(box1[0], box2[0])  # 交集左上角x
        x2 = min(box1[2], box2[2])  # 交集右下角x
        y1 = max(box1[1], box2[1])  # 交集左上角y
        y2 = min(box1[3], box2[3])  # 交集右下角y

        overlap = max(0., x2 - x1) * max(0., y2 - y1)
        # union = (box1[2]-box1[0]) * (box1[3]-box1[1]) \
        #         + (box2[2]-box2[0]) * (box2[3]-box2[1]) \
        #         - overlap

        return overlap

    def IoU(self, bbox):
        overlap = self.overlap(bbox)
        return overlap / (self.area() + bbox.area() - overlap)

    def proportion(self, bbox, style="area"):
        assert (style in ["area", "width", "height"])
        if style == "area":
            overlap = self.overlap(bbox)
            return overlap / self.area()
        elif style == "height":
            x1 = max(self.get()[0], bbox.get()[0])  # 交集左上角x
            x2 = min(self.get()[2], bbox.get()[2])  # 交集右下角x
            y1 = max(self.get()[1], bbox.get()[1])  # 交集左上角y
            y2 = min(self.get()[3], bbox.get()[3])  # 交集右下角y
            return max(0., y2 - y1) / abs(self.get()[3] - self.get()[1])
        elif style == "width":
            x1 = max(self.get()[0], bbox.get()[0])  # 交集左上角x
            x2 = min(self.get()[2], bbox.get()[2])  # 交集右下角x
            y1 = max(self.get()[1], bbox.get()[1])  # 交集左上角y
            y2 = min(self.get()[3], bbox.get()[3])  # 交集右下角y
            return max(0., x2 - x1) / abs(self.get()[2] - self.get()[0])

    def intersection(self, bbox):

        if self.IoU(bbox) > 0:
            inter = copy.deepcopy(self)
            inter.x1 = max(self.get()[0], bbox.get()[0])  # 交集左上角x
            inter.x2 = min(self.get()[2], bbox.get()[2])  # 交集右下角x
            inter.y1 = max(self.get()[1], bbox.get()[1])  # 交集左上角y
            inter.y2 = min(self.get()[3], bbox.get()[3])  # 交集右下角y
            return inter
        else:
            return None

    def union(self, bbox):
        union = copy.deepcopy(self)
        union.x1 = min(self.get()[0], bbox.get()[0])  # 交集左上角x
        union.x2 = max(self.get()[2], bbox.get()[2])  # 交集右下角x
        union.y1 = min(self.get()[1], bbox.get()[1])  # 交集左上角y
        union.y2 = max(self.get()[3], bbox.get()[3])  # 交集右下角y
        return union

    #     def random_move(self):
    #         bbox_w, bbox_h= self.x2-self.x1,self.y2-self.y1
    #         img_w,img_h=self.img_size.get()
    #         x = int(random.uniform(0, img_w - bbox_w))
    #         y = int(random.uniform(0, img_h - bbox_h))
    #         self.x1,self.y1=x,y
    #         self.x2,self.y2=self.x1+bbox_w,self.y1+bbox_h
    #         return self

    def offset(self, x, y):
        self.x1 += x
        self.x2 += x
        self.y1 += y
        self.y2 += y

    #     def move_by_center(self,center):
    #         center_x,center_y=center[0],center[1]

    #         bbox_w, bbox_h= self.x2-self.x1,self.y2-self.y1
    #         x = center_x-bbox_w//2
    #         y = center_y-bbox_h//2

    #         img_w,img_h=self.imgsz.get()

    #         x= min(max(x,0),img_w-bbox_w)
    #         y= min(max(y,0),img_h-bbox_h)

    #         self.x1,self.y1=x,y
    #         self.x2,self.y2=self.x1+bbox_w,self.y1+bbox_h

    # def _bbox2mask(self,bbox, img_size, foreground=255):
    #     """
    #     bbox  : (x1,y1,x2,y2)
    #     img_size: (w,h)
    #     """
    #     mask = np.zeros((img_size[1], img_size[0]), np.uint8)
    #     x1, y1, x2, y2 = bbox
    #     # print(bbox)
    #     mask[y1:y2, x1:x2] = foreground
    #     return mask

    def clone(self):
        return copy.deepcopy(self)


    def pad_and_random_move(self,min_size):
   
        
        half_max_side= max((self.x2-self.x1),(self.y2-self.y1))//4*3
        min_size= (max(min_size[1],half_max_side),max(min_size[1],half_max_side))
        
        new_w=my_truncat(self.x2-self.x1,min_size[0],self.imgsz.w)
        # print(self.y2-self.y1,min_size[1],self.imgsz.h)|
        new_h = my_truncat(self.y2 - self.y1, min_size[1], self.imgsz.h)
        # print(new_w,new_h)
        x1_range= [max(0, self.x2-new_w ), min(self.x1,self.imgsz.w-new_w ) ]
        x1_range[1]= max(x1_range[0]+1,x1_range[1])
        y1_range = [max(0, self.y2 - new_h), min(self.y1, self.imgsz.h - new_h)]
        y1_range[1] = max(y1_range[0] + 1, y1_range[1])
        x1, y1= random.randint(int(x1_range[0]),int(x1_range[1])),random.randint(int(y1_range[0]),int(y1_range[1]))
        x2,y2= x1+new_w ,y1+new_h
        new_bbox=self.clone()
        new_bbox.x1=x1
        new_bbox.y1=y1
        new_bbox.x2=x2
        new_bbox.y2=y2
        return new_bbox

    def crop_by_roi(self,roi):
        assert self.IoU(roi)>0
        new_bbox=self.clone()
        new_bbox.x1= new_bbox.x1- roi.x1
        new_bbox.y1 = new_bbox.y1 - roi.y1
        new_bbox.x2 = new_bbox.x2 - roi.x1
        new_bbox.y2 = new_bbox.y2 - roi.y1
        new_bbox.x1=my_truncat(new_bbox.x1,0,roi.Width())
        new_bbox.x2=my_truncat(new_bbox.x2,0,roi.Width())
        new_bbox.y1=my_truncat(new_bbox.y1,0,roi.Height())
        new_bbox.y2=my_truncat(new_bbox.y2,0,roi.Height())
        return new_bbox


import json
class JsonData(object):

    def __init__(self, json_data=None):

        if json_data is not None:
            assert (isinstance(json_data, (dict, str)))
            if isinstance(json_data, str):
                self.load_from(json_data)
            else:
                self.json_data = json_data

    def load_from(self, json_path: str):
        assert (os.path.exists(json_path)), json_path
        with open(json_path, "r", encoding="utf-8") as f:
            self.json_data = json.load(f)
        return self

    def get_llm_shapes(self):
        return self.json_data["shapes"]

    def if_has_label(self, cate):
        # print(self.json_data)
        for i in range(len(self.json_data["shapes"])):
            shape = self.json_data["shapes"][i]
            if shape["label"] == cate:
                return True
        return False


    def get_llm_imginfo(self):
        return {"imageName":self.json_data["imagePath"],"imageData":self.json_data["imageData"],
                "imageHeight":self.json_data["imageHeight"],"imageWidth":self.json_data["imageWidth"]}


    def save_to(self,json_path):
        with open(json_path, "w",encoding="utf-8") as f:
            json.dump(self.json_data, f, ensure_ascii=False, indent=2)
            
            
            
            
def get_time_str():
    import datetime
    from datetime import datetime
    now=datetime.now()
    return now.strftime("%m/%d/%Y, %H:%M:%S")


def get_suffix(stri):
    return stri.split(".")[-1]


            
        
from torch.utils.data import Dataset
import random

class BalancedDataWrapper(Dataset):
    def __init__(self, dataset,key):
        self.dataset = dataset
        self.label_to_indices = {}
        self.key=key
        # 创建一个从label到样本索引的映射
        for idx, sample in enumerate(dataset.sample_by_instance):
            label = sample[self.key]
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        self.labels = list(self.label_to_indices.keys())
        
        # 计算所有类别的平均样本数
        total_samples = sum(len(indices) for indices in self.label_to_indices.values())
        self.samples_per_label = total_samples // len(self.labels)
        self.idx2label=[ self.labels[idx % len(self.labels)] for idx  in range(len(self.dataset)) ]
        
    def __getitem__(self, idx):
        label = self.idx2label[idx]
        indices = self.label_to_indices[label]
        sample_idx = random.choice(indices)  # 随机从该类别的indices里面选取
        return self.dataset[sample_idx]
    
    def __len__(self):
        return len(self.dataset)
        # return len(self.labels) * self.samples_per_label
        
        

from torch.utils.data import Dataset
class DatasetWrapper(Dataset):
    
    def __init__(self,dataset, mapping=None,new_length=None,transform=None):
        
        self.dataset=dataset
        self.mapping=mapping
        self.new_length=new_length
        self.transform=transform
        
        if self.new_length is not None:
            self._iidxes=[ i%len(self.dataset) for i  in range(self.new_length)]
        else:
            self._iidxes=[ i%len(self.dataset) for i  in range(len(self.dataset))]
            
    def __len__(self):
        return len(self._iidxes)
    
    def __getitem__(self, idx):
        idx=self._iidxes[idx]
        item=self.dataset[idx]
        if self.transform is not None:
            item["img"]=self.transform(item["img"])
        if self.mapping==None:
            pass
        elif  isinstance(self.mapping,tuple) or isinstance(self.mapping,list):
            item= [item[k] for k in self.mapping]
        elif  isinstance(self.mapping,dict):  
            item= { k: item[v] for k, v in self.mapping.items() } 
        else:
            item=item[self.mapping]
        return item 
    
    def get_sample_wights(self,key):
        from collections import Counter
        all_value= [ sam[key] for sam in self ]
        # print(self.cls_dict)
        value_num=dict( Counter(all_value))
        num_total=len(self)
        weights=[]
        # print(len(self))
        for data in self:
            w=num_total/value_num[data[key]]
            weights.append(w)
        return weights

    

class BaseSampleProcess(object):
    
    def __call__(sample):
        return sample