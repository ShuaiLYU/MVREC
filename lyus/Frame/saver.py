
import os
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import torch
import logging
import  time


from pylab import mpl

def is_scalar(value):
    try:
        value =float(value)
    except:
        return False
    return True


class Saver(object):
    def __init__(self,save_dir,**kwargs):
        self.save_dir=save_dir
        if not os.path.exists(save_dir):
            print("make dir :{}".format(save_dir))
            os.makedirs(save_dir)
        ###生成logger
        self.__logger=self.get_logger(save_dir)
        ###生成writer
        self.__writer=self.get_writer(save_dir)
        ###生成visual
        self.__visualer=self.get_visualer(save_dir)
        self.input_size=kwargs.get("input_size",(1,3,256,256))

    # 打印日志到控制台和log_path下的txt文件
    def get_logger(slef,log_dir):
        # log_path = log_path + "/Log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timer = time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime())
        logger = logging.getLogger(log_dir)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(levelname)s]   %(asctime)s		%(message)s')
        consoleHandel = logging.StreamHandler()
        consoleHandel.setFormatter(formatter)
        logger.addHandler((consoleHandel))
        txthandle = logging.FileHandler((log_dir + '/' + timer + 'log.txt'))
        txthandle.setFormatter(formatter)
        logger.addHandler(txthandle)
        return logger

    def get_writer(self,save_dir):
        log_dir = os.path.join(save_dir,"tb_log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return SummaryWriter(log_dir)


    def get_visualer(self,save_dir):
        photos_dir = os.path.join(save_dir,"visualization")
        if not os.path.exists(photos_dir):
            os.makedirs(photos_dir)
        return Visualer(photos_dir)

    def info(self,msg):
        self.__logger.info(msg)

    def visualize(self, batch_list, child_dir, file_name_batch,**kwargs):
        self.__visualer.visualize(batch_list, child_dir, file_name_batch,**kwargs)

    def add_scalar(self,tag, scalar_value, global_step=None, walltime=None):
        if(not is_scalar(scalar_value)):
            print("[warning], {} check scalar failed!, discard... ".format(type(scalar_value)))
            return 
        self.__writer.add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        self.__writer.add_scalars(main_tag, tag_scalar_dict, global_step=global_step, walltime=walltime)

    def add_garph(self, model, input_to_model=None, verbose=False):
        if input_to_model is None:
            input_to_model=torch.randn(*self.input_size)
        self.__writer.add_graph( model, input_to_model=input_to_model, verbose=False)
        if verbose:
            #hether to print graph structure in console.
            self.info(model)

    def close(self):
        self.__writer.close()

import collections
class Visualer(object):

    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            print("create  directory:{} ".format(save_dir))
            os.makedirs(save_dir)

    def visualize(self, batch_list, child_dirs, file_name_batch,**kwargs):


        file_names = [file_name_batch[i] for i in range(len(file_name_batch))]
        if isinstance(child_dirs, (list,tuple)):
            save_dirs=[  os.path.join(self.save_dir, dir) for  dir  in child_dirs ]
        else:
            save_dirs= os.path.join(self.save_dir, child_dirs)
        # print(   self.save_dir,save_dirs)
        self._visualize(batch_list, file_names, save_dirs)



    def _visualize(self,list_batchs, filenames, save_dirs):
        """
        :param list_batchs:  list[  array[b,h,w,c] ,array[b,h,w,c] ]  or  [   [imags],[images]  ]
        :param filenames:        list [filename]
        :param save_dir:
        :return:
        """
        # batch_num= filenames.shape(0)
        for i, filename in enumerate(filenames):
            if not isinstance(save_dirs, list):
                # save_dirs = [save_dirs for i in len(batch_num)]
                save_dir = save_dirs
            else:
                save_dir = save_dirs[i]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if not isinstance(filename, str): filename = filename.decode("utf-8")
            filename = filename.replace("/", "_")
            list_images = []
            for batchs in list_batchs:
                image = np.array(batchs[i])
                if len(image.shape) > 2 and image.shape[2] == 1: image = image.squeeze(2)
                list_images.append(image)
            img_visual = self.concatImage(list_images, offset=10)
            visualization_path = os.path.join(save_dir, filename)
            try:
                img_visual.save(visualization_path)
            except:
                print("图片保存失败【[]】".format(visualization_path))

    def concatImage(self, images, mode="Adapt", scale=0.5, offset=None):
        """
        :param images:  图片列表
        :param mode:     图片排列方式["Row" ,"Col","Adapt"]
        :param scale:
        :param offset:    图片间距
        :return:
        """
        if not isinstance(images, list):
            raise Exception('images must be a  list  ')
        if mode not in ["Row", "Col", "Adapt"]:
            raise Exception('mode must be "Row" ,"Adapt",or "Col"')
        images = [np.uint8(img) for img in images]  # if Gray  [H,W] else if RGB  [H,W,3]
        images = [img.squeeze(2) if len(img.shape) > 2 and img.shape[2] == 1 else img for img in images]
        count = len(images)
        img_ex = Image.fromarray(images[0])
        size = img_ex.size  # [W,H]
        if mode == "Adapt":
            mode = "Row" if size[0] <= size[1] else "Col"
        if offset is None: offset = int(np.floor(size[0] * 0.02))
        if mode == "Row":
            target = Image.new(img_ex.mode, (size[0] * count + offset * (count - 1), size[1] * 1), 100)
            for i in range(count):
                image = Image.fromarray(images[i]).resize(size, Image.BILINEAR).convert(img_ex.mode)
                target.paste(image, (i * (size[0] + offset), 0))
                # target.paste(image, (i * (size[0] + offset), 0, i * (size[0] + offset) + size[0], size[1]))
            return target
        if mode == "Col":
            target = Image.new(img_ex.mode, (size[0], size[1] * count + offset * (count - 1)), 100)
            for i in range(count):
                image = Image.fromarray(images[i]).resize(size, Image.BILINEAR).convert(img_ex.mode)
                target.paste(image, (0, i * (size[1] + offset)))
                # target.paste(image, (0, i * (size[1] + offset), size[0], i * (size[1] + offset) + size[1]))
            return target


