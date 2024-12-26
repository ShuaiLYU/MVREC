
import os
import json

import  csv
import  pandas  as pd

import numpy as np

import PIL.Image as Image
import cv2




def list_folder(root,use_absPath=True, func=None):
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
            path = os.path.join(cul_dir, fname)#.replace('\\', '/')
            if  func is not None and not func(path):
                continue
            if use_absPath:
                files.append(path)
            else:
                files.append(os.path.relpath(path,root))
    print("    find {} file under {}".format(len(files), root))
    return files


def is_image(filename):
    if filename.endswith(".jpg"):
        return True
    elif filename.endswith(".bmp"):
        return True
    elif filename.endswith(".png"):
        return True
    else:
        return False

def is_json(filename):
    if filename.endswith(".json"):
        return True
    else:
        return False

def remove_suffix(filename):
    return ".".join(filename.split(".")[:-1])

def find_images_and_labels(root,use_abspath=True,is_img=None,label_suffix=".json"):
    if is_img is None:
        is_img=is_image
    imgpaths=list_folder(root,use_abspath,is_img)
    labelpathset=list_folder(root,use_abspath,lambda x:x.endswith(label_suffix))

    samples=[]
    samples_wo_label=[]
    for imgpath in imgpaths:
        labelpath=remove_suffix(imgpath)+label_suffix
        if labelpath in labelpathset:
            samples.append((imgpath,labelpath))

        else:
            samples_wo_label.append(imgpath)
    print("[INfo] found {}  matched samples, {} samples without labels".format(len(samples),len(samples_wo_label)))
    return samples


def load_json_file(filename):
    assert os.path.isfile(filename) and is_json(filename)
    return json.load(open(filename, encoding='utf-8'))


#load single labels for imagepatch with labelme format
def load_single_labelme_shape(filename):
    shapes=load_json_file(filename)["shapes"]
    # assert(len(shapes)==1),"find {} shapes unexpectedly in [{}]".format(len(shapes),filename)
    return shapes[0]

"""
samples 是一个 list( sample1,sample2, sample3,... )  each sample:  dict{key1:val1,key2:val2}
"""

class CsvLabelTool(object):
    
    @staticmethod
    def sample_list_to_csv(samples,csvname=None):
        assert isinstance(samples,list) and len(samples)>0
        for sam in samples:
            assert  isinstance(sam,dict)
        print(samples[0])
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
    @staticmethod
    def csv_to_sample_list(csvname):
        data=pd.read_csv(csvname)
        return data.apply(pd.Series.to_dict, axis=1).to_list()


    @staticmethod
    def write_data_config_to_csv(config_dir, dataset_dict):
        """
        :param dir:    写入目标文件夹
        :param dataset_dict:   {"train" :  list1[]...,"valid": ..., "test":,,,}
        :param shuffle:  是否打乱列表
        :return:
        """
        if not os.path.exists(config_dir): os.makedirs(config_dir)

        for key,val in dataset_dict.items():

            if len(val)==0: continue
            csv_name=os.path.join(config_dir,key+".csv")
            CsvLabelTool.sample_list_to_csv(val,csv_name)

            print(u"数据集配置保存到:【{}】".format(config_dir))

    import random


class LabelConfigTool():
    
    @staticmethod
    def divide(cls_dict, train_ratio, valid_ratio, shuffle=True):
        """
        :param cls_dict:   {  0：[exampels],1:[exampels] }
        :param train_ratio:   0.75
        :param valid_ratio:    0.1
        :param shuffle:
        :return:
        """
        if (train_ratio + valid_ratio) > 1:
            raise Exception("wrong params...")
        print(u"划分数据集......")
        ratio_dict = {"train": train_ratio, "valid": valid_ratio, "test": 1 - train_ratio - valid_ratio}
        dataset_dict = {key: [] for key, val in ratio_dict.items()}
        for cls, data_list in cls_dict.items():
            sample_num = len(data_list)
            train_offset = int(np.floor(sample_num * ratio_dict["train"]))
            val_offset = int(np.floor(sample_num * (ratio_dict["train"] + ratio_dict["valid"])))
            print (u" 类别[{}]中，训练集：{}，验证集：{}，测试集：{}" \
                 .format(cls, train_offset, val_offset - train_offset, len(data_list) - val_offset))
            Keys = ["train"] * train_offset \
                   + ["valid"] * (val_offset - train_offset) \
                   + ["test"] * (len(data_list) - val_offset)
            if shuffle:
                random.shuffle(data_list)
            for key, item in zip(Keys, data_list):
                dataset_dict[key].append(item)
        return dataset_dict


    @staticmethod
    def categorize_samples(samples,key="label"):
        cls_dict={}
        for sam in samples:
            # print(sam)
            label=sam[key]
            if label not in cls_dict.keys():
                cls_dict[label]=[]
            cls_dict[label].append(sam)
        return cls_dict
    @staticmethod
    def  divide_samples(samples, train_ratio, valid_ratio, shuffle=True, key="label"):
        cls_dict=LabelConfigTool.categorize_samples(samples,"label")
        return LabelConfigTool.divide(cls_dict,train_ratio,valid_ratio,shuffle)


def gen_dataset_config(dataset,save_dir,config_name,train_ratio,valid_ratio,shuffle=True,key="label"):
    samples=[ sam for sam in dataset]
    sample_parts=LabelConfigTool.divide_samples(samples,train_ratio,valid_ratio,shuffle,key)
    config_dir=os.path.join(save_dir,config_name)
    CsvLabelTool.write_data_config_to_csv(config_dir,sample_parts)
    
    

def get_file_dir(filename):

    return  os.path.abspath(os.path.dirname(filename))


 
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    img=np.array(img)
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)





import random
def random_crop(img, bbox, prob):
    if random.random() > prob:
        return img
    else:
        w, h = img.size
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w - 1), min(y2, h - 1)
        
        new_bbox = (random.randint(0, x1), random.randint(0, y1), random.randint(x2, w - 1), random.randint(y2, h - 1))
        if (new_bbox[2]>new_bbox[0] and  new_bbox[3]>new_bbox[1]):
            return img.crop(new_bbox)
        else:
            return img





if __name__ =="__main__":

    img_dir=r"F:\FABRIC_DATASET\classfication_dataset_part1\2022-4-11_TAL湖蓝色梭织布_PC1F1\飞毛\sF2_C2_exp70.00_sp5.0_2022411_14_37_31_674_38.98_42.63_1943_(663,925,791,1024).jpg"
    img=Image.open(img_dir)
    print(img.size)
    save_dir=r"C:\Users\shuai\Desktop\temp\New folder"
    for i in range(100):
        img_crop=random_crop(img,[53,39,75,59],0.8)
        img_crop.save(save_dir+"/"+str(i)+".jpg")

