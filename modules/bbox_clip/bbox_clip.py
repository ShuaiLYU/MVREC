import clip
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

from pathlib import Path

__all__ = ['PatchClip']

import numpy as np

from .bbox_tool import BboxTool
from .atten_pool import AttentionPool2D
from .feature_extractor import FeatureExtractor
from .utils import deepCopyTorchLayer
import torch.nn as nn





def calculate_iou(bbox1, bbox2):
    # 计算交集区域的坐标
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # 计算交集区域的面积
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # 如果交集区域的面积为0，返回False
    if intersection_area == 0:
        return False

    # 计算两个矩形的面积
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # 计算并集区域的面积
    union_area = bbox1_area + bbox2_area - intersection_area

    # 计算交并比
    iou = intersection_area / union_area

    return iou > 0


def check_iou_greater_than_zero(roi_list, roi_bboxes):
    assert len(roi_list) == len(roi_bboxes), "Lists must have the same length"

    result = []
    for roi, bbox in zip(roi_list, roi_bboxes):

        # result.append(calculate_iou(roi, bbox))
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if not calculate_iou([0, 0, 9, 9], bbox) or area < 1:
            pass
        print(bbox, area)

    # print(result)


def monitor_updates(threshold=10):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # 初始化或获取之前存储的状态
            if not hasattr(self, 'prev_weight'):
                self.prev_weight = None
                self.unchanged_count = 0

            # 调用原始的forward方法
            result = func(self, *args, **kwargs)

            # 获取指定层的权重（这里假设是第一个nn.Linear层）
            current_weight = self.adapter[0].weight.detach()

            # 检查权重是否更新
            if self.prev_weight is not None:
                if torch.equal(current_weight, self.prev_weight):
                    self.unchanged_count += 1
                else:
                    self.unchanged_count = 0
            self.prev_weight = current_weight.clone()

            # 检查是否达到了停止的阈值
            assert self.unchanged_count < threshold, "Parameter has not changed for 10 consecutive times."

            return result  # 直接返回结果
        return wrapper
    return decorator

class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim):
        super().__init__()
        self.adapter  = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, img_embedding):
        # Save original shape for reshaping later
        original_shape = img_embedding.shape
      
        assert original_shape[-1]==self.input_dim

        # If the input tensor has more than 2 dimensions, flatten it to 2D
        if img_embedding.dim() > 2:
            img_embedding = img_embedding.view(-1, self.input_dim)  # Flatten keeping the last dimension
        
        # Pass through the adapter
        adapted = self.adapter(img_embedding)
        if img_embedding.dim() > 2:
            new_shape = original_shape[:-1] + (self.output_dim,)
            adapted = adapted.view(new_shape)
        
        return adapted


class Classification(nn.Module):
    def __init__(self, input_size, class_num):
        super().__init__()
        self.classification_head = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.LeakyReLU(),
            nn.Linear(input_size // 2, class_num)
        )
        self.classification_head.apply(self.init_bert_weights)

    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, img_embedding):
        return self.classification_head(img_embedding)

from clip import model
class BboxClip(nn.Module):
    def __init__(self,input_size=512, backbone='RN50',**kwargs):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert  backbone in ['RN50', 'RN50x4', 'RN50x16','ViT-B/16','ViT-B/32']
        self.feat_dim= {"RN50": 2048, "RN50x4":2560,"RN50x16":3072,'ViT-B/16':1024,'ViT-B/32':1024}[backbone]
        self.input_size=input_size
        self.backbone_stride=32
        # 这里应该使用 atten_pool的预训练 shape ?
        self.roi_shape = (self.input_size // self.backbone_stride, self.input_size // self.backbone_stride)

        self.init_from_clip(clip.load(backbone, self.device)[0])

        self.with_adapter = kwargs.get("with_adapter", False)
        self.with_classification = kwargs.get("with_classification", False)

        # fc_feat_dim=640
        # if self.with_adapter == True:
        #     self.image_adapter = Adapter(fc_feat_dim, 320)

        # if self.with_classification == True:
        #     self.classification = Classification(fc_feat_dim, 24)

    def init_from_clip(self, clip_model):
        self.layers_to_extract_from = ["layer4"]  # "layer2",
        self.visual = clip_model.visual
        self.model = FeatureExtractor( self.visual , self.layers_to_extract_from)
        self.aggre_features_func = lambda x: x[self.layers_to_extract_from[0]]
   

        self.token_embedding = clip_model.token_embedding



        self.positional_embedding = clip_model.positional_embedding

        self.ln_final = clip_model.ln_final

        self.text_projection = clip_model.text_projection
        self.logit_scale = deepCopyTorchLayer(clip_model.logit_scale)
        self.transformer = clip_model.transformer


        # self.pool2D=CropPool2D()
        # self.roi_shape = (clip_model.visual.input_resolution // 32, clip_model.visual.input_resolution // 32)
        # self.pool2D=clip_model.visual.attnpool
        clip_model.visual.attnpool.embed_dim=self.feat_dim  # 123
        self.pool2D = AttentionPool2D( self.roi_shape[0],clip_model.visual.attnpool)
        # self.pool2D=model.AttentionPool2d(     self.roi_shape[0],clip_model.visual.attnpool.embed_dim,
        #                                        clip_model.visual.attnpool.num_heads,
        #                                        clip_model.visual.attnpool.output_dim)




                

    def encode_text(self, text):

        with torch.no_grad():
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


    def get_image_feat(self, img_tensor):
        # with torch.no_grad():
        # img_tensor = torch.stack([self.preprocess(img).to(self.device) for img in imgList])
        # img_tensor = self.preprocess_batch(imgList).to(self.device)  # Make sure tensor is on the right device
        # print(torch.isnan(img_tensor).any(),img_tensor.max(),img_tensor.min(),img_tensor.shape,3242)
        features = self.model(img_tensor)
        # for _,f in features.items():      print(torch.isnan(f).any())
        # print(torch.isnan(features).any(),2232)
        aggre_feature = self.aggre_features_func(features)
        # print(torch.isnan(aggre_feature).any(),aggre_feature.shape,22432)
        # print(aggre_feature)
        return aggre_feature

    def encode_image(self, image_tensor, bboxes: torch.tensor):
        # with torch.no_grad():
        assert len(bboxes.shape) == 3


        # print(bboxes,bboxes.shape)
        def _get_roi_feats(img_feats, roi_bboxes):
            assert len(img_feats.shape) == 4
            assert img_feats.shape[0] == len(roi_bboxes)
            roi_feats = []
            # with torch.no_grad():
            for i, feat_bbox in enumerate(roi_bboxes):
                x1_f, y1_f, x2_f, y2_f = feat_bbox
                img_feat = img_feats[i]
                roi_feat = img_feat[:, y1_f:y2_f, x1_f:x2_f]
                roi_feats.append(roi_feat)
            return torch.stack(roi_feats, dim=0)
        with torch.no_grad(): #  ???
            image_feats = self.get_image_feat(image_tensor)

            batch = image_feats.shape[0]
            img_shape = (image_tensor.shape[2], image_tensor.shape[3])
            feat_shape = image_feats.shape[2:]
            # print(feat_shape)
            # roi_list, roi_bboxes = zip(
            #     *[BboxTool.cal_roi_bbox(bbox[0].tolist(), img_shape, feat_shape, self.roi_shape) for bbox in bboxes])
            roi_shape=self.pool2D.get_input_shape2D()
            print(roi_shape)
            # roi_shape=feat_shape
            roi_list, roi_bboxes = zip(
                *[BboxTool.cal_roi_bbox(bbox[0].tolist(), img_shape, feat_shape, roi_shape) for bbox in bboxes]) # ???
            
            # check_iou_greater_than_zero(roi_list,roi_bboxes)
            # roi_bboxes = BboxTool.shrink_to_center_pixel(roi_bboxes)
            self.img_shape = img_shape
            self.feat_shape = feat_shape
            self.roi_list = roi_list
            self.roi_bboxes = roi_bboxes
            roi_feats = _get_roi_feats(image_feats, roi_list)       
            pool_roi_feats = self.pool2D(roi_feats, roi_bboxes)

            # pool_roi_feats= self.pool2D(roi_feats)
            # pool_roi_feats /= pool_roi_feats.norm(dim=-1, keepdim=True)
            # if self.with_adapter == True:
            #     pool_roi_feats = self.image_adapter(pool_roi_feats)

        return pool_roi_feats

    # def forward(self, image, bboxes, text):
    #     image_features_ori = self.encode_image(image, bboxes)
    #     text_features_ori = self.encode_text(text)

    #     # ------------------adapter--------------------


    #     if self.with_classification == True:
    #         classification_feature = self.classification(image_features_ori)

    #     # normalized features
    #     image_features = image_features / image_features.norm(dim=1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=1, keepdim=True)

    #     # cosine similarity as logits
    #     logit_scale = self.logit_scale.exp()
    #     logits_per_image = logit_scale * image_features @ text_features.t()
    #     logits_per_text = logits_per_image.t()

    #     # shape = [global_batch_size, global_batch_size]
    #     if self.with_classification == True:
    #         return logits_per_image, logits_per_text, classification_feature
    #     else:
    #         return logits_per_image, logits_per_texts

    def tokenize(self, caption, truncate=False):
        return clip.tokenize(caption, truncate=truncate)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def save(self, path):
        """保存模型到指定路径"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        torch.save(self.state_dict(), path)
        print(f'Model saved to {path}')

    def load(self, path):
        """从指定路径加载模型"""
        path = Path(path)
        self.load_state_dict(torch.load(path))
        self.to(self.device)  # 确保模型在正确的设备上
        self.eval()  # 设置为评估模式，如果你打算进行推理
        print(f'Model loaded from {path}')

