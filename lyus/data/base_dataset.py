from PIL import Image
import numpy as np
import logging
import os
import cv2
import random

EXTENSIONS= ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']

def has_file_allowed_extension(filename, extensions=EXTENSIONS):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

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




def divide_dataset(cls_dict, train_ratio, val_ratio, shuffle=True):
    """
    :param cls_dict:   {  0：[exampels],1:[exampels] }
    :param train_ratio:   0.75
    :param val_ratio:    0.1
    :param shuffle:
    :return:
    """
    if (train_ratio + val_ratio) > 1:
        raise Exception("wrong params...")
    print(u"划分数据集......")
    ratio_dict = {"training": train_ratio, "validation": val_ratio, "testing": 1 - train_ratio - val_ratio}
    dataset_dict = {key: [] for key, val in ratio_dict.items()}
    for cls, data_list in cls_dict.items():
        sample_num = len(data_list)
        train_offset = int(np.floor(sample_num * ratio_dict["training"]))
        val_offset = int(np.floor(sample_num * (ratio_dict["training"] + ratio_dict["validation"])))
        # print (u" 类别[{}]中，训练集：{}，验证集：{}，测试集：{}" \
        #     .format(cls, train_offset, val_offset - train_offset, len(data_list) - val_offset))
        Keys = ["training"] * train_offset \
               + ["validation"] * (val_offset - train_offset) \
               + ["testing"] * (len(data_list) - val_offset)
        if shuffle:
            random.shuffle(data_list)
        for key, item in zip(Keys, data_list):
            dataset_dict[key].append(item)
    return dataset_dict

def write_txt(dir, dataset_dict, prefix='%s %s %d', shuffle=False, clear=True):
    """
    :param dir:    写入目标文件夹
    :param dataset_dict:   {"training" :  list1[]...}
    :param prefix:      写入格式
    :param shuffle:  是否打乱列表
    :param clear:  是否清空之间的写入对象
    :return:
    """
    if not os.path.exists(dir): os.makedirs(dir)
    mode = 'w' if clear else 'a'
    for key, sample_list in dataset_dict.items():  # 每个类别
        if shuffle: random.shuffle(sample_list)
        write_path = os.path.join(dir, key) + '.txt'
        with open(write_path, 'a', encoding='utf-8') as file:
            lines = [(prefix % (image[0], image[1], image[2])) for image in sample_list]
            file.write('\n'.join(lines))
    print(u"数据集配置保存到txt:【{}】".format(dir))

def read_txt(data_dir,phase_re=None):
    """Read the content of the text file and store it into lists."""
    phases = ["training", "validation", "testing",None]
    assert phase_re in phases
    assert os.path.exists(data_dir),"{}".format(data_dir)
    data_dict = {phase: [] for phase in phases[:-1]}
    for phase, data_list in data_dict.items():
        txt_file = data_dir + "/" + phase + ".txt"
        if not os.path.exists(txt_file):
            continue
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                items = list(line.strip().split(' '))
                items=[item  if item !="None" else None for item in items ]
                data_list.append(items)
        data_dict[phase] = data_list
    if phase_re ==None:
        return data_dict
    else:
        return data_dict[phase_re]


class BDataset(object):
    def __init__(self,root):
        self.root=root
        self.samples=[]
        self.cls_dict={}
    def list_dir(self,root,use_absPath, func):
        return  list_folder(root,use_absPath,func)

    def  make_dataset(self):
        raise Exception("")

    def split_dataset_by_cls(self,samples,loc):
        """
        :param samples:   samples= [ sample1, sample2, ... ]
        :param loc:     the location index of class in sample
        :return:
        """
        # 根据类别生成字典
        cls_dict = {}
        for sample in samples:
            cls = int(sample[loc])
            if cls not in cls_dict.keys():
                cls_dict[cls] = []
            cls_dict[cls].append(sample)
        for cls, sample_list in cls_dict.items():
            print("    类别{}，样本数为：{}".format(cls, len(sample_list)))
        return cls_dict

    def sort_samples(self,cls_dict):
        # #####从新排列样本 [ class1,class2,class3,class1,class2,class3,...]
        samples=[]
        print(cls_dict)
        max_len=max([ len(cls_dict[i]) for i  in range(len(cls_dict)) ])
        for i  in range(max_len):
            for j in range(len(cls_dict)):
                if i<len(cls_dict[j]):
                    samples.append(cls_dict[j][i])
        return  samples

    def gen_a_sample(self,sample):
        raise Exception("")
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
#         print(len(self),idx)
        sample = self.samples[idx]
        return self.gen_a_sample(sample)

    def getitem_cls(self, idx, cls):
        sample = self.cls_dict[cls][idx]
        return self.gen_a_sample(sample)

class BaseDataset(object):
    def __init__(self,Datadir,ConfigDir):
        self.Datadir=os.path.abspath(Datadir)
        self.ConfigDir=os.path.abspath(ConfigDir)
        self.list_folder(self.Datadir)
        self.sample_list,self.cls_dict=self.make_dataset()
        self.Config=os.path.exists(ConfigDir)

    def make_dataset(self, useAbsDir=False):
        print("生成数据集......")
        root=self.Datadir
        samples=[]
        #添加样本列表
        for img in sorted(self.imgs):
            label = img.split(".")[-2] + "_label.bmp"
            # 过滤没有语义标签的图片
            if label not in self.imgs_pixel:
                continue
            # 过滤不符合类别的图片
            target = 1 if np.sum(cv2.imread(label)) > 10 else 0
            if not useAbsDir:
                img=os.path.relpath(img,root).replace('\\','/')
                label=os.path.relpath(label,root).replace('\\','/')
            item = (img, label, target)
            samples.append(item)
        print("    总样本数：{}".format(len(samples)))
        #根据类别生成字典
        cls_dict={}
        for sample in samples:
            cls=sample[2]
            if cls  not in cls_dict.keys():
                cls_dict[cls]=[]
            cls_dict[cls].append(sample)
        for cls,sample_list in cls_dict.items():
            print("    类别{}，样本数为：{}".format(cls,len(sample_list)))
        return samples,cls_dict

    @staticmethod
    def list_folder(root,extensions=EXTENSIONS):
        return list_folder(root,has_file_allowed_extension)
    @staticmethod
    def divide_dataset(cls_dict, train_ratio, val_ratio, shuffle=True):
        return divide_dataset(cls_dict, train_ratio, val_ratio, shuffle)
    @staticmethod
    def write_txt(dir, dataset_dict, prefix='%s %d %d', shuffle=False, clear=True):
        return write_txt(dir, dataset_dict, prefix, shuffle, clear)
    @staticmethod
    def read_txt(data_dir):
        return read_txt(data_dir)
    @staticmethod
    def has_file_allowed_extension(filename, extensions=EXTENSIONS):
        return has_file_allowed_extension(filename, extensions)


