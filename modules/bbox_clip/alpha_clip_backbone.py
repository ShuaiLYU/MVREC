import torch 
import torch.nn  as nn 
import alpha_clip
from torchvision import transforms
import numpy as np
import os 
import torch.nn.functional as F
from .utils import deepCopyTorchLayer




class AlphaClip(nn.Module):
    def __init__(self,input_size=224,backbone="ViT-L/14" ):
        super(AlphaClip,self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert backbone in  ["ViT-L/14", "ViT-B/16", "ViT-L/14@336px"]
        if backbone=="ViT-L/14":
            alpha_vision_ckpt_pth="/home/lyushuai/Projects/AlphaCLIP/checkpoints/clip_l14_grit20m_fultune_2xe.pth"
            assert os.path.exists(alpha_vision_ckpt_pth)
            self.backbone, self.preprocess = alpha_clip.load(backbone,alpha_vision_ckpt_pth=alpha_vision_ckpt_pth, 
                                                            device=self.device)  # change to your own ckpt path
        if backbone=="ViT-B/16": # 
            alpha_vision_ckpt_pth="/home/lyushuai/Projects/AlphaCLIP/checkpoints/clip_b16_grit20m_fultune_2xe.pth"
            assert os.path.exists(alpha_vision_ckpt_pth)
            self.backbone, self.preprocess = alpha_clip.load(backbone,alpha_vision_ckpt_pth=alpha_vision_ckpt_pth, 
                                                            device=self.device)  # change to your own ckpt path

        if backbone=="ViT-L/14@336px":
            alpha_vision_ckpt_pth="/home/lyushuai/Projects/AlphaCLIP/checkpoints/clip_l14@336_grit_20m_4xe.pth"
            assert os.path.exists(alpha_vision_ckpt_pth)
            self.backbone, self.preprocess = alpha_clip.load(backbone,alpha_vision_ckpt_pth=alpha_vision_ckpt_pth, 
                                                            device=self.device)  # change to your own ckpt path        

        self.input_size=input_size
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((224, 224)), # change to (336,336) when using ViT-L/14@336px
            transforms.Normalize(0.5, 0.26)])
        
    def tokenize(self,text_list):
        text=alpha_clip.tokenize(text_list) 
        return text 
        


    def encode_text(self, text):

        return self.backbone.encode_text(text)

    def copy_logit_scale_module(self):
        
        return deepCopyTorchLayer(self.backbone.logit_scale)





    def encode_image(self,image_tensor,masks): #9 15 

        from lyus.Frame import Experiment
        k_patch=int(Experiment().get_param().debug.k_patch)
        alpha_list=masks

        # return  self.backbone.visual(image_tensor, alpha_list)
        image_features = self.visual_forward(self.backbone.visual,image_tensor, alpha_list)
        # return image_features
         # torch.Size([81, 1, 768]) torch.Size([81, 256, 768])
        global_features,local_features=image_features[:,:1,:],image_features[:,1:,:]

        # assert False
        if k_patch<1:
            return global_features
        else:
            cosine_sim = F.cosine_similarity(global_features, local_features, dim=-1)
            # print(global_features.shape, local_features.shape, cosine_sim.shape)
            # assert False
            # Sort the local_features based on the cosine similarity
            knn=k_patch
            # sorted_indices = torch.argsort(cosine_sim, dim=1, descending=True)[:,:knn]  # Get indices for sorting
            sorted_logits, sorted_indices = torch.sort(cosine_sim, descending=True, dim=1)
            sorted_logits, sorted_indices = sorted_logits[:,:knn], sorted_indices[:,:knn]
            # Reorder local_features using sorted indices
            # Use torch.arange to create batch indices and sorted_indices to index along the second dimension
            batch_indices = torch.arange(local_features.size(0)).unsqueeze(1).repeat(1, knn)
            sorted_local_features = local_features[batch_indices, sorted_indices]
            assemble_features=torch.cat([global_features,sorted_local_features], dim=1)#.mean(dim=1,keepdim=True)
            # assemble_weights= torch.cat([ torch.ones_like(sorted_logits[:,:1]), sorted_logits], dim=1).unsqueeze(-1)
            # assemble_features=torch.sum(assemble_features *assemble_weights,dim=1,keepdim=True)/ torch.sum(assemble_weights,dim=1,keepdim=True)
            return assemble_features
        # assert False
        # _,loc =torch.max(cosine_sim,dim=-1)
        # # global_features = (global_features+local_features[torch.arange(local_features.size(0)), loc].unsqueeze(1) )/2
   

        # scaled_bboxes = (bboxes / 14)
        # # Apply ceil to x1 and y1 (first and second columns)
        # scaled_bboxes[:,:, 0] = torch.floor(scaled_bboxes[:,:, 0])
        # scaled_bboxes[:,:, 1] = torch.floor(scaled_bboxes[:,:, 1])
        # # Apply floor to x2 and y2 (third and fourth columns)
        # scaled_bboxes[:,:, 2] = torch.ceil(scaled_bboxes[:,:, 2])
        # scaled_bboxes[:,:, 3] = torch.ceil(scaled_bboxes[:,:, 3])


        def create_masks_from_bboxes(bboxes_scaled, grid_size=16):

            n = bboxes_scaled.shape[0]
            # Create an empty mask of size [n, grid_size, grid_size]
            masks = torch.zeros((n, grid_size, grid_size), dtype=torch.float32,device=bboxes_scaled.device)
            # Loop through each bounding box to create masks
            for idx, (x1, y1, x2, y2) in enumerate(bboxes_scaled):
                # Convert coordinates to integer for indexing
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Use slicing to fill the area of the bounding box
                masks[idx, y1:y2, x1:x2] = 1
            # Reshape masks to [n, grid_size * grid_size]
            return masks.view(n, -1).bool()
        masks=create_masks_from_bboxes(scaled_bboxes.squeeze(1),16)
   
        def cul_roi_feat_vect(x, masks):
            
            masks= masks.unsqueeze(-1)
            # 使用key_padding_mask更新x的值，将需要忽略的位置设置为NaN
            x = torch.where(masks, x, torch.tensor(float('nan'), device=x.device))

            # 沿着序列维度计算平均值，忽略NaN
            x_mean = torch.nanmean(x, dim=1,keepdim=True)
            # print(x_mean)
            return x_mean
        # print(local_features.shape, masks.shape)
        # assert False
        mean_local_features = cul_roi_feat_vect(local_features, masks)
   
    
        # print( global_features.shape,local_features.shape,masks.shape,mean_local_features.shape)

        # assert False
        #  image_features   torch.Size([81, 257, 768])  # B, L ,C 

        # print(torch.norm(image_features, p=2, dim=1))
        return global_features

    @staticmethod
    def visual_forward(self, x: torch.Tensor, alpha=None, return_attn=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # ASSUME alpha is always not None!
        x = x + self.conv1_alpha(alpha)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        if return_attn:
            x, attn_last = self.transformer(x, return_attn=True)
        else:
            x = self.transformer(x, return_attn=False)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # print(x.shape)    #torch.Size([81, 257, 1024])
        # x = self.ln_post(x[:, 0, :])

        def forward2d(module, x):
        # x is expected to be a 3D tensor of shape [batch, seq_length, features]
            # Save original shape for reshaping later
            original_shape = x.size()
            x = x.view(-1, original_shape[2])
            # Apply LayerNorm or any other operation that expects 2D input
            x = module(x)
            # Reshape back to the original 3D shape [batch, seq_length, features]
            x = x.view(original_shape)
            return x
        x = forward2d(self.ln_post,x)
 
        # print(torch.unique(alpha))  #tensor([-1.9229,  1.9229],
      
        if self.proj is not None:  
            x = x @ self.proj   #  torch.Size([81, 257, 768])  # B, L ,C 

        # print(x.shape, alpha.shape)
        # assert False
    
        if return_attn:
            return x, attn_last
        else:
            return x



class AlphaClipWomask(AlphaClip):

    def encode_image(self,image_tensor,masks): #9 15 

        max_value = masks.max()

        # 用最大值填充 masks
        masks.fill_(max_value)
        return AlphaClip.encode_image(self,image_tensor,masks)

