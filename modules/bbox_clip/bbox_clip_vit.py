import clip 
import torch
import torch.nn as nn 
import torch.nn.functional as F
from .bbox_tool import BboxTool
from modules.bbox_clip.bbox_clip import BboxClip, deepCopyTorchLayer

class BboxClipVit(BboxClip):
    def __init__(self, input_size=512, backbone='ViT-B/16', **kwargs):
        super().__init__(input_size, backbone, **kwargs)  # Properly calling the superclass __init__
        BboxClipVit.resize_positional_embedding(self.visual,self.input_size)


    def init_from_clip(self, clip_model):
        self.visual = clip_model.visual
        self.model = self.visual
        # print(1111111111111111111111)
        self.aggre_features_func = lambda x: x[self.layers_to_extract_from[0]]
   

        self.token_embedding = clip_model.token_embedding

        self.positional_embedding = clip_model.positional_embedding

        self.ln_final = clip_model.ln_final

        self.text_projection = clip_model.text_projection
        self.logit_scale = deepCopyTorchLayer(clip_model.logit_scale)
        self.transformer = clip_model.transformer


    @staticmethod
    def resize_positional_embedding(self, input_resolution: int, mode: str = "bicubic"):
        src_shape=self.positional_embedding.shape
        if input_resolution != self.input_resolution:
            Nt = self.input_resolution // self.patch_size
            nNt = input_resolution // self.patch_size
            class_positional_embedding = self.positional_embedding.data[[0], :]
            image_positional_embedding = self.positional_embedding.data[1:, :]
            image_positional_embedding = image_positional_embedding.unsqueeze(0).permute(0, 2, 1)
            B, D, L = image_positional_embedding.shape
            image_positional_embedding = image_positional_embedding.reshape(B, D, Nt, Nt)
            image_positional_embedding = F.interpolate(
                image_positional_embedding, size=(nNt, nNt), mode=mode, align_corners=False,
            )
            image_positional_embedding = image_positional_embedding.squeeze(0).view(D, -1).permute(1, 0)
            self.positional_embedding = nn.Parameter(
                torch.cat([class_positional_embedding, image_positional_embedding], dim=0),
                requires_grad=False,
            )
            self.input_resolution = input_resolution
        print(f"self.positional_embedding.shape {src_shape} -> {self.positional_embedding.shape}  ")


    
    def create_key_padding_mask_from_bboxes(self, bboxes, H, W):
        def create_key_padding_mask_from_roi(roi, H, W):

                key_padding_mask = torch.ones((H, W), dtype=torch.bool)  # Start with everything masked
           
                x1, y1, x2, y2 = map(int, roi.squeeze())
                key_padding_mask[y1:y2, x1:x2] = False  # Unmask the ROI
                # for i in range(key_padding_mask.shape[0]):
                #     print(i,key_padding_mask[i])
                return key_padding_mask.view(-1)
        if bboxes is not None:
            key_padding_masks = [create_key_padding_mask_from_roi(bbox, H, W) for bbox in bboxes]
            # Add an unmasked value for the mean vector
            key_padding_masks = [torch.cat([torch.tensor([False]), mask]) for mask in key_padding_masks]
            key_padding_mask = torch.stack(key_padding_masks, dim=0).to(bboxes.device)

        else:
            key_padding_mask = None
        return key_padding_mask


    def encode_image(self, x: torch.Tensor,bboxes):
        # return self.visual(x)

        def cul_roi_feat_vect(x_in, key_padding_mask):
            # 移除第一行，因为原始函数中对每个样本都执行了 x[:,i,:][1:]
            x = x_in[1:, :,: ]
            key_padding_mask = key_padding_mask[:, 1:].transpose(1,0).unsqueeze(2)
        
            # 使用key_padding_mask更新x的值，将需要忽略的位置设置为NaN
            x = torch.where(key_padding_mask, torch.tensor(float('nan'), device=x.device), x)

            # 沿着序列维度计算平均值，忽略NaN
            x_mean = torch.nanmean(x, dim=0)
            x_in[0]=x_mean
            return x_in

        def residualAttentionBlock_forward(residualAttentionBlock, 
                                           x: torch.Tensor,key_padding_mask:torch.Tensor=None,
                                           with_roi_feat=False):

            atten_x= residualAttentionBlock.ln_1(x)  # [num_patch+1,batch,num_feat]


            atten_x=cul_roi_feat_vect(atten_x,key_padding_mask=key_padding_mask)

            atten_x=residualAttentionBlock.attn(atten_x,atten_x, atten_x, need_weights=False, key_padding_mask=None)[0]
            x = x +atten_x

         
            # print(x.shape, key_padding_mask.shape)
            x = x + residualAttentionBlock.mlp(residualAttentionBlock.ln_2(x))
            return x

        
        def transformer_forward(transformer, x: torch.Tensor,key_padding_mask):
            xs=[]
            for index, residualAttentionBlock in enumerate(transformer.resblocks):

                # x=residualAttentionBlock(x)
                # key_padding_mask=None
       
             
                if index < len(transformer.resblocks)-3 :
                    x =residualAttentionBlock_forward(residualAttentionBlock,x,key_padding_mask,True)
                else:
             
                    x =residualAttentionBlock_forward(residualAttentionBlock,x,key_padding_mask,True)
                    xs.append(x)
            # return torch.mean(torch.stack(xs),dim=0,keepdim=False)
            return x
                

  


            return x
        
        def VisionTransformer_forward(visionTransformer, x: torch.Tensor,bboxes):
            x = visionTransformer.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([visionTransformer.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + visionTransformer.positional_embedding.to(x.dtype)
            x = visionTransformer.ln_pre(x)


            x = x.permute(1, 0, 2)  # NLD -> LND
            h,w=self.model.input_resolution // self.model.patch_size,self.model.input_resolution // self.model.patch_size

            img_shape=[self.model.input_resolution,self.model.input_resolution]
            feat_shape=[h,w]
     
            _, roi_bboxes = zip(
                *[BboxTool.cal_roi_bbox(bbox[0].tolist(), img_shape, feat_shape, feat_shape) for bbox in bboxes])
            roi_bboxes=torch.tensor(list(roi_bboxes)).unsqueeze(1).to(bboxes.device)
            key_padding_mask=self.create_key_padding_mask_from_bboxes(roi_bboxes,h,w)
     
            x = transformer_forward(visionTransformer.transformer,x,key_padding_mask=key_padding_mask)
     
            x = x.permute(1, 0, 2)  # LND -> NLD

            x = visionTransformer.ln_post(x[:, 0, :])

            if visionTransformer.proj is not None:
                x = x @ visionTransformer.proj
            return x
        with torch.no_grad():
            return VisionTransformer_forward(self.visual,x,bboxes=bboxes)
    

# if __name__ == "__main__":
#     # 初始化模型
#     model = BboxClipVit(512)

#     # 创建一个假的4维图像张量 [batch_size, channels, height, width]
#     # 例如：batch_size = 2, channels = 3 (RGB), height = 224, width = 224
#     img_tensor = torch.rand(2, 3, 512, 512)

#     # 创建一个对应的3维边界框张量 [batch, 1, 4]
#     # 边界框的格式可能是 [x1, y1, x2, y2]，其中值通常在图像尺寸范围内
#     bboxes = torch.tensor([[[10, 10, 100, 100]], [[50, 50, 150, 150]]], dtype=torch.float32)

#     # 使用模型对图像和边界框进行编码
#     features = model.encode_image(img_tensor, bboxes)

#     # 打印输出的特征
#     print(features)
