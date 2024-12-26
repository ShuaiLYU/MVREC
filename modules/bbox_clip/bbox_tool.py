
import numpy as np 
import os 
class BboxTool:
    
    def __init__(self):
        pass
    
    @staticmethod
    def pad_bbox(array_shape, bbox, target_shape):
        array_height,array_width = array_shape
        x1, y1, x2, y2 = bbox
        bbox_height, bbox_width = y2 - y1, x2 - x1

        target_height, target_width = target_shape

        # 计算需要填充或裁剪的大小
        pad_crop_height = target_height - bbox_height
        pad_crop_width = target_width - bbox_width

        # 计算新的边界框坐标，确保它们不超出图像边界
        new_x1 = max(0, x1 - pad_crop_width // 2)
        new_y1 = max(0, y1 - pad_crop_height // 2)

        # 如果新的边界框加上填充或裁剪的宽度/高度超过图像的宽度/高度，则调整新的边界框坐标
        new_x1 = min(new_x1, array_width - (target_width))
        new_y1 = min(new_y1, array_height - (target_height))

        new_x2 = new_x1 + target_width
        new_y2 = new_y1 + target_height

        return (new_x1, new_y1, new_x2, new_y2)
    @staticmethod
    def rescale_bbox(bbox,src_shape, dst_shape):
        # 解包输入
        x1, y1, x2, y2 = bbox
        h1, w1 = dst_shape
        h,w = src_shape
        # 计算特征图上的边界框坐标
        x1_f = int(np.floor(x1 * w1 / w))
        y1_f = int(np.floor(y1 * h1 / h))
        x2_f = int(np.ceil(x2 * w1 / w))
        y2_f = int(np.ceil(y2 * h1 / h))

        return x1_f,y1_f,x2_f,y2_f
    
    @staticmethod
    def crop_by_roi(bbox, roi):
        bbox1=[min(max(bbox[0],roi[0]),roi[2]),
               min(max(bbox[1],roi[1]),roi[3]),
              min(max(bbox[2],roi[0]),roi[2]),
              min(max(bbox[3],roi[1]),roi[3])]

        x1,y1,x2,y2=bbox1[0]-roi[0],bbox1[1]-roi[1],bbox1[2]-roi[0],bbox1[3]-roi[1]
        return [x1,y1,x2,y2]
    @staticmethod
    def cal_roi_bbox(bbox,img_shape,feat_shape,roi_shape):
        feat_bbox=BboxTool.rescale_bbox(bbox,img_shape,feat_shape)
        roi=BboxTool.pad_bbox(feat_shape,feat_bbox,roi_shape)
        # print(img_shape,feat_shape,feat_bbox,roi)
        roi_bbox=BboxTool.crop_by_roi(feat_bbox,roi)
        if roi_bbox[2]==roi_bbox[0] or roi_bbox[1]==roi_bbox[3]:

            print(bbox,img_shape,feat_shape,roi_shape,roi_bbox)
            roi_bbox=[roi_bbox[0],roi_bbox[1],roi_bbox[2],roi_bbox[3]]
            if roi_bbox[2]==roi_bbox[0]:roi_bbox[2]=roi_bbox[0]+1
            if roi_bbox[3]==roi_bbox[1]:roi_bbox[3]=roi_bbox[1]+1
        return roi,roi_bbox
    @staticmethod   
    def shrink_to_center_pixel(roi_bboxes):
        # 初始化一个空列表来存储新的边界框
        center_pixel_bboxes = []

        # 遍历输入列表中的每个边界框
        for bbox in roi_bboxes:
            # 计算边界框的中心坐标
            x_center = (bbox[0] + bbox[2]) // 2
            y_center = (bbox[1] + bbox[3]) // 2

            # 创建一个新的边界框，只包括中心像素
            center_pixel_bbox = [x_center, y_center, x_center + 1, y_center + 1]

            # 将新的边界框添加到结果列表中
            center_pixel_bboxes.append(center_pixel_bbox)

        return center_pixel_bboxes