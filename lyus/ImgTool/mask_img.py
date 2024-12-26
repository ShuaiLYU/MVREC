
from pathlib import Path
from PIL import Image, ImageDraw
import random
import numpy as np

class Mask(object):

    def __init__(self, mask, label_classes=None, mode="P"):

        self.mask = np.array(mask)
        self.label_classes = label_classes

        palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]])
        self.palette = np.concatenate([np.zeros((1, 3)), palette], axis=0).astype(np.uint8)

    def if_has_ojbects(self):

        if np.max(self.mask) > 0:
            return True
        else:
            return False

    def save(self, mask_path):

        mask = Image.fromarray(self.mask).convert("P")
        mask.putpalette(self.palette)
        mask.save(mask_path)

    def crop(self, bbox):
        mask_img = Image.fromarray(self.mask)
        crop_mask = Mask(mask_img.crop(bbox), label_classes=self.label_classes)
        return crop_mask


    def to_bin(self,foreground:int=-1):
        if foreground==-1:
            mask = np.where(np.array(self.mask) > np.zeros_like(self.mask), 1, 0).astype(np.uint8)
        else:
            mask = np.where(np.array(self.mask)==foreground, 1, 0).astype(np.uint8)
        return Mask(mask,label_classes=self.label_classes)

import json
from labelme.utils import  shapes_to_label



from LyusLib.Image.mask_img import Mask

import json
from labelme.utils import  shapes_to_label

class LabelmeMask(Mask):
    def __init__(self,jsonfile,label_name_to_value):
        mask=self.get_mask_from_labelme_json(jsonfile,label_name_to_value)
        super(LabelmeMask,self).__init__(mask,label_name_to_value)
    def get_mask_from_labelme_json(self,json_file, label_name_to_value):
        data = json.load(open(json_file, encoding='utf-8'))
        img_shape = (data["imageHeight"], data["imageWidth"])
        cls_mask, _ = shapes_to_label(img_shape, data["shapes"], label_name_to_value)

        return cls_mask






def get_random_mask(img_size, mask_size, rotation=20):
    img = Image.new("L", img_size, 0)
    mask = Image.new("L", mask_size, 255)
    mask = mask.rotate(rotation, expand=True)
    center_x = random.randint(0, img_size[0] - mask_size[0])
    center_y = random.randint(0, img_size[1] - mask_size[1])

    bbox = [center_x, center_y, center_x + mask.width, center_y + mask.height]
    img.paste(mask, box=bbox)
    return img

