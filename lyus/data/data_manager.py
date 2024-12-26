import os
from random import shuffle

import os
from random import shuffle
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from utils import utils
from utils.data import my_transforms
from utils.data.gen_defect import DefectiveGenerator

class DataManager(object):
    def __init__(self, dataList,param,shuffle=True):
        """
        """
        self.shuffle=shuffle
        self.data_list=dataList
        self.data_size=len(dataList)
        self.sample_dict =self.splitData(self.data_list)
        self.data_dir=param["data_dir"]
        self.epochs_num=param["epochs_num"]
        self.batch_size = param["batch_size"]
        self.image_scale=param["image_scale"]
        self.image_size =param["image_size"]
        self.with_RGB=image_size =param["with_RGB"]
        self.image_size = [self.image_size[0] // self.image_scale, self.image_size[1] // self.image_scale]
        self.set()
    def set(self):
        self.number_batch =len(self.data_list)//self.batch_size
        self.next_batch=self.get_next()

    def get_next(self):
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.int32,tf.int32, tf.string))
        dataset = dataset.repeat(self.epochs_num)
        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size*3+200)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch

    def generator(self):
        while True:
            for index in range(len(self.data_list)):
                yield self.get_one_sample(self.data_list[index])

    def get_one_sample(self,sample):
        file_basename_image, file_basename_label, label =sample
        image_path = os.path.join(self.data_dir, file_basename_image)
        label_path = os.path.join(self.data_dir, file_basename_label)
        image = self.read_data(image_path)
        label_pixel = self.read_data(label_path)
        label_pixel = self.label_preprocess(label_pixel)
        if not self.with_RGB:
            image = (np.array(image[:, :, np.newaxis]))
        label_pixel = (np.array(label_pixel[:, :, np.newaxis]))
        image = utils.transform(image)
        return image, label_pixel, int(label), file_basename_image

    def __iter__(self):
        for index in range(self.number_batch):
            next_batch=SESSION.run(self.next_batch)
            yield next_batch

    def read_data(self, data_name):
        flag=1 if self.with_RGB else 0
        img = cv2.imread(data_name, flag)  # /255.#read the gray image
        img = cv2.resize(img, (int(self.image_size[1]), int(self.image_size[0])))
        return img

    def label_preprocess(self,label):
        #label = cv2.resize(label, (int(self.image_size[1]/8), int(self.image_size[0]/8)))
        label_pixel=self.ImageBinarization(label)
        return  label_pixel

    def ImageBinarization(self,img, threshold=1):
        img = np.array(img)
        image = np.where(img > threshold, 1, 0)
        return image

    def splitData(self,data):
        """
        把数据列表按照类别分开
        :param data:
        :return:
        """
        dict={}
        for item in data:
            key=int(item[2])
            if key not in dict.keys():
                dict[key]=[]
            dict[key].append(item)
        return dict

class DataManager_balance(DataManager):
    def __init__(self, dataList,param,shuffle=True):
        super(DataManager_balance,self).__init__(dataList,param,shuffle)
        self.with_transform=param["with_transform"]
        transform_train=[
                my_transforms.GroupRandomHorizontalFlip(),
                my_transforms.GroupRandomVerticalFlip(),
                #transforms.RandomResizedCrop
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomResizedCrop(size=[], scale=(0.5, 1.0),ratio=()),
                # transforms.RandomCrop()
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        if "with_rotate" in param.keys() and param["with_rotate"]:transform_train.append(my_transforms.GroupRandomRotation())
        self.transform = {"train":transforms.Compose(transform_train), "val": None}
    def set(self):
        self.next_batch = self.get_next()
        self.number_batch=int(np.floor(len(self.sample_dict[1])))*2//self.batch_size

    def get_next(self):
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.int32, tf.int32, tf.string))
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch

    def generator(self):
        step=0
        while(True):
            for cls, sample_list, in self.sample_dict.items():
                sample_num=len(sample_list)
                index=step%sample_num
                if index==0 and self.shuffle:
                    shuffle(sample_list)
                yield self.get_one_sample(sample_list[index])
            step+=1
    #添加transform
    def get_one_sample(self,sample):
        file_basename_image, file_basename_label, label =sample
        image_path = os.path.join(self.data_dir, file_basename_image)
        label_path = os.path.join(self.data_dir, file_basename_label)
        image = self.read_data(image_path)
        label_pixel = self.read_data(label_path)
        if self.with_transform:
            image,label_pixel=self.transform_sample(image,label_pixel)
        label_pixel = self.label_preprocess(label_pixel)
        image = utils.transform(image)
        if not self.with_RGB:
            image = (np.array(image[:, :, np.newaxis]))
        label_pixel = (np.array(label_pixel[:, :, np.newaxis]))
        return image, label_pixel, int(label), file_basename_image

    def transform_sample(self,image,label):
        image = Image.fromarray( np.uint8(image))
        label = Image.fromarray(np.uint8(label))
        ouput=self.transform["train"]([image,label])
        image=np.array(ouput[0])
        label = np.array(ouput[1])
        return image,label

class DataManager_normal(DataManager):
    def __init__(self, dataList,param,shuffle=True):
        super(DataManager_normal,self).__init__(dataList,param,shuffle)
        self.transform = {"train":
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.RandomResizedCrop(size=[], scale=(0.5, 1.0),ratio=()),
                # transforms.RandomCrop()
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "val": None}
    def set(self):
        self.next_batch = self.get_next()
        self.number_batch=int(np.floor(len(self.sample_dict[1])))

    def get_next(self):
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.int32, tf.int32, tf.string))
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch

    def generator(self):
        step=0
        while(True):
            for cls, sample_list, in self.sample_dict.items():
                if cls==0:
                    sample_num=len(sample_list)
                    index=step%sample_num
                    if index==0 and self.shuffle:
                        shuffle(sample_list)
                    yield self.get_one_sample(sample_list[index])
            step+=1
    #添加transform
    def get_one_sample(self,sample):
        file_basename_image, file_basename_label, label =sample
        image_path = os.path.join(self.data_dir, file_basename_image)
        label_path = os.path.join(self.data_dir, file_basename_label)
        image = self.read_data(image_path)

        label_pixel = self.read_data(label_path)
        label_pixel = self.label_preprocess(label_pixel)
        image = (np.array(image[:, :, np.newaxis]))
        label_pixel = (np.array(label_pixel[:, :, np.newaxis]))
        image,label_pixel=self.transform_sample(image,label_pixel)
        image = utils.transform(image)
        return image, label_pixel, int(label), file_basename_image

    def transform_sample(self,image,label):
        image= np.uint8(image)
        label = np.uint8(label)
        img=np.concatenate((image,label),2)
        img = Image.fromarray(img)
        img=self.transform["train"](img)
        img=np.array(img)
        image=img[:,:,0][:, :, np.newaxis]
        label =img[:, :, 1][:, :, np.newaxis]
        return image,label


class DataManager_faker(DataManager):
    def __init__(self, dataList,param,dir_DefectsDir,shuffle=True):
        super(DataManager_faker,self).__init__(dataList,param,shuffle)
        self.defectGenerator=DefectiveGenerator(dir_DefectsDir,self.image_size,[0,10000])
    def set(self):
        self.next_batch = self.get_next()
        self.number_batch=len(self.sample_dict[0])//self.batch_size*2

    def get_next(self):
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.int32, tf.int32, tf.string))
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch

    def generator(self):
        step=0
        while(True):
            sample_list = self.sample_dict[0]
            sample_num = len(sample_list)
            index = step % sample_num
            if index == 0 and self.shuffle:
                shuffle(sample_list)
            sample = self.get_one_sample(sample_list[index])
            step += 1
            for cls in range(2):
                if cls==0:
                    yield sample
                if cls==1:
                    yield self.draw_one_sample(sample)

    def draw_one_sample(self,sample):
        image, label_pixel, label, file_basename_image=sample
        image=image.squeeze(2)
        image_draw, label_pixel_draw = self.defectGenerator.genDefect(image)
        image_draw = (np.array(image_draw[:, :, np.newaxis]))
        label_pixel_draw = (np.array(label_pixel_draw[:, :, np.newaxis]))
        filename = str(file_basename_image).split(".")[-2]+"_faker."+str(file_basename_image).split(".")[-1]
        return image_draw, label_pixel_draw, int(1), filename


class DataManager_class(DataManager):
    def __init__(self, dataList,param,dir_DefectsDir,shuffle=True):
        super(DataManager_class,self).__init__(dataList,param,shuffle)
        self.defectGenerator=DefectiveGenerator(dir_DefectsDir,self.image_size,[0,10000])
    def set(self):
        # self.number_batch_negative =int(np.floor(len(self.sample_dict[0])/self.batch_size))
        # self.number_batch_positive =int(np.floor(len(self.sample_dict[1])/self.batch_size))
        self.number_batch = int(np.floor(len(self.sample_dict[1]) / self.batch_size))
        self.next_batch_positive=self.get_next_positive()
        self.next_batch_negative=self.get_next_negative()
    def get_next_positive(self):
        dataset = tf.data.Dataset.from_generator(self.generator_positive, (tf.float32, tf.int32,tf.int32, tf.string))
        dataset = dataset.repeat(self.epochs_num*3)
        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size*3+200)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch
    def generator_negative(self):
        while True:
            data_list=self.sample_dict[0]
            for index in range(len(data_list)):
                yield  self.get_one_sample(data_list[index])

    def get_next_negative(self):
        dataset = tf.data.Dataset.from_generator(self.generator_negative, (tf.float32, tf.int32,tf.int32, tf.string))
        dataset = dataset.repeat(self.epochs_num*3)
        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size*3+200)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch

    def generator_positive(self):
        while True:
            data_list=self.sample_dict[0]
            for index in range(len(data_list)):
                yield self.draw_one_sample(self.get_one_sample(data_list[index]))

    def draw_one_sample(self,sample):
        image, label_pixel, label, file_basename_image=sample
        image=image.squeeze(2)
        image_draw, label_pixel_draw = self.defectGenerator.genDefect(image)
        image_draw = (np.array(image_draw[:, :, np.newaxis]))
        label_pixel_draw = (np.array(label_pixel_draw[:, :, np.newaxis]))
        return image_draw, label_pixel_draw, int(1), file_basename_image

if __name__=="__main__":

    kolektorSDD_Patch_config="../configs/kolektorSDD_config1"
