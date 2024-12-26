
import torch
import torch.nn.functional as F
from torch import nn
from .utils import deepCopyTorchLayer


class CropPool2D(torch.nn.Module):

    def __init__(self):
        super(CropPool2D, self).__init__()
    def forward(self,img_feats,bboxes):
        assert len(img_feats.shape) == 4
        assert img_feats.shape[0] == len(bboxes)
        roi_feats=[]
        # with torch.no_grad():

        for i,feat_bbox in enumerate(bboxes):
            x1_f,y1_f, x2_f,y2_f=feat_bbox
            img_feat=img_feats[i]
            roi_feat=img_feat[:, y1_f:y2_f, x1_f:x2_f]
            roi_feat_mean=roi_feat.mean(dim=(1,2)).squeeze().unsqueeze(0)
            roi_feats.append(roi_feat_mean)
        return torch.concat(roi_feats,dim=0)

from clip import  model


  


class AttentionPool2D(nn.Module):
    def __init__(self, spacial_dim, attenPool2D:model.AttentionPool2d):
        super().__init__()
        # self.positional_embedding = deepCopyTorchLayer(attenPool2D.positional_embedding)
        embed_dim=attenPool2D.embed_dim

        self.k_proj = deepCopyTorchLayer(attenPool2D.k_proj)
        self.q_proj =deepCopyTorchLayer( attenPool2D.q_proj)
        self.v_proj = deepCopyTorchLayer(attenPool2D.v_proj)
        self.c_proj =deepCopyTorchLayer( attenPool2D.c_proj)



        self.num_heads = attenPool2D.num_heads
        # self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.positional_embedding =deepCopyTorchLayer(attenPool2D.positional_embedding)
        # self.resize_positional_embedding(spacial_dim)

        # learned    k  v , q , ，c , positional_embedding
        # k: 虚席 


    def get_input_shape2D(self):
        old_len=int((self.positional_embedding.shape[0]-1) ** 0.5)
        return (old_len,old_len)

    def create_key_padding_mask_from_roi(self, roi, H, W):

        key_padding_mask = torch.ones((H, W), dtype=torch.bool)  # Start with everything masked
        x1, y1, x2, y2 = map(int, roi)
        key_padding_mask[y1:y2, x1:x2] = False  # Unmask the ROI
        # for i in range(key_padding_mask.shape[0]):
        #     print(key_padding_mask[i])
        return key_padding_mask.view(-1)

    def resize_positional_embedding(self, input_resolution: int, mode: str = "bicubic"):
        # print(self.positional_embedding.shape,222)
        old_len=int((self.positional_embedding.shape[0]-1) ** 0.5)
        if input_resolution !=old_len:
            Nt = input_resolution 
            nNt = input_resolution
            #  Nt = input_resolution // self.patch_size
            # nNt = input_resolution // self.patch_size

            class_positional_embedding = self.positional_embedding.data[[0], :]
            image_positional_embedding = self.positional_embedding.data[1:, :]
            image_positional_embedding = image_positional_embedding.unsqueeze(0).permute(0, 2, 1)
            B, D, L = image_positional_embedding.shape
            # image_positional_embedding = image_positional_embedding.reshape(B, D, Nt, Nt)
            image_positional_embedding = image_positional_embedding.reshape(B, D, old_len, old_len)
            image_positional_embedding = F.interpolate(
                image_positional_embedding, size=(nNt, nNt), mode=mode, align_corners=False,
            )
            image_positional_embedding = image_positional_embedding.squeeze(0).view(D, -1).permute(1, 0)
            self.positional_embedding = nn.Parameter(
                torch.cat([class_positional_embedding, image_positional_embedding], dim=0),
                requires_grad=False,
            )
            self.input_resolution = input_resolution
        print("self.positional_embedding.shape= ", self.positional_embedding.shape)


    def get_roi_feat(self,img_feats,bboxes):
        assert len(img_feats.shape) == 4
        assert img_feats.shape[0] == len(bboxes)
        roi_feats=[]
        # if torch.isnan(img_feats).any():
        #     print(torch.isnan(img_feats).any(),222)
        # with torch.no_grad():
        for i,feat_bbox in enumerate(bboxes):
            x1_f,y1_f, x2_f,y2_f=feat_bbox
            img_feat=img_feats[i]
            roi_feat=img_feat[:, y1_f:y2_f, x1_f:x2_f]
            # roi_feat=img_feat
            roi_feat_mean=roi_feat.mean(dim=(1,2)).squeeze().unsqueeze(0)
            if torch.isnan(roi_feat_mean).any():
                print(torch.isnan(img_feat).any())
                print(feat_bbox)
            roi_feats.append(roi_feat_mean)

        roi_feats=torch.concat(roi_feats,dim=0)    
        if torch.isnan(roi_feats).any():
            print(torch.isnan(img_feats).any())
            print(bboxes)
            assert(False)
            
        return roi_feats

    def forward(self, x, bboxes=None):
        # print(x.shape,self.positional_embedding.shape)
        _, _, H, W = x.shape
        if bboxes is not None:
            key_padding_masks = [self.create_key_padding_mask_from_roi(bbox, H, W) for bbox in bboxes]
            # Add an unmasked value for the mean vector
            key_padding_masks = [torch.cat([torch.tensor([False]), mask]) for mask in key_padding_masks]
            key_padding_mask = torch.stack(key_padding_masks, dim=0).to(x.device)

        else:
            key_padding_mask = None

  
        roi_feat=self.get_roi_feat(x,bboxes).unsqueeze(0) # 1NC

        feat=self._forward_func(x,key_padding_mask,roi_feat)
        return feat

    def _forward_func(self,x, key_padding_mask, roi_feat):
        # roi_feat=x.mean(dim=0, keepdim=True)
        # key_padding_mask=None
        # print(roi_feat.shape)

        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([roi_feat, x], dim=0)  # (HW+1)NC
        # repalce the mean of x  with the mean  roi region
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC


        query = x[:1]
        key = x
        value = x
        x, _ = F.multi_head_attention_forward(
            query=query, key=key, value=value,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
            key_padding_mask=key_padding_mask
        )
        # print(x.shape)
        if torch.isnan(x).any():
            print(torch.isnan(query).any(),torch.isnan(key).any(),torch.isnan(value).any())
            print(query.shape,key.shape,value.shape,111)
            assert(False)
        return x.squeeze(0)


