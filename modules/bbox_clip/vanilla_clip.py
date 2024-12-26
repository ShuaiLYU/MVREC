
import clip

import torch 
from .utils import deepCopyTorchLayer
import torch.nn as nn 
import torch.nn.functional as F
class VanillaClip(nn.Module):
    def __init__(self,input_size=512, backbone='RN50',**kwargs):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # assert  backbone in ['RN50', 'RN50x4', 'RN50x16','ViT-B/16','ViT-B/32']
        # self.feat_dim= {"RN50": 2048, "RN50x4":2560,"RN50x16":3072,'ViT-B/16':1024,'ViT-B/32':1024}[backbone]
        self.input_size=input_size
        # self.backbone_stride=32

        self.backbone,  self.preprocess =clip.load(backbone, self.device)

    def tokenize(self,text_list):
        text=clip.tokenize(text_list) 
        return text 

    def encode_text(self, text):

        return self.backbone.encode_text(text)
    
    def copy_logit_scale_module(self):
        
        return deepCopyTorchLayer(self.backbone.logit_scale)


    # def encode_image(self,image_tensor,masks): #9 15 
    #     # assert False

    #     return self.backbone.encode_image(image_tensor)

    def encode_image(self,image_tensor,masks): #9 15 

        # from lyus.Frame import Experiment
        # k_patch=int(Experiment().get_param().debug.k_patch)

        # return  self.backbone.visual(image_tensor, alpha_list)
        # image_features = self.visual_forward(self.backbone.visual,image_tensor, alpha_list)
        image_features=self.backbone.encode_image(image_tensor)
        #  # torch.Size([81, 1, 768]) torch.Size([81, 256, 768])
        image_features= image_features.unsqueeze(1)
        print(image_features.shape)
        return image_features

        # global_features,local_features=image_features[:,:1,:],image_features[:,1:,:]

        # # assert False
        # if k_patch<1:
        #     return global_features
        # else:
        #     cosine_sim = F.cosine_similarity(global_features, local_features, dim=-1)
        #     # print(global_features.shape, local_features.shape, cosine_sim.shape)
        #     # assert False
        #     # Sort the local_features based on the cosine similarity
        #     knn=k_patch
        #     # sorted_indices = torch.argsort(cosine_sim, dim=1, descending=True)[:,:knn]  # Get indices for sorting
        #     sorted_logits, sorted_indices = torch.sort(cosine_sim, descending=True, dim=1)
        #     sorted_logits, sorted_indices = sorted_logits[:,:knn], sorted_indices[:,:knn]
        #     # Reorder local_features using sorted indices
        #     # Use torch.arange to create batch indices and sorted_indices to index along the second dimension
        #     batch_indices = torch.arange(local_features.size(0)).unsqueeze(1).repeat(1, knn)
        #     sorted_local_features = local_features[batch_indices, sorted_indices]
        #     assemble_features=torch.cat([global_features,sorted_local_features], dim=1)#.mean(dim=1,keepdim=True)
        #     # assemble_weights= torch.cat([ torch.ones_like(sorted_logits[:,:1]), sorted_logits], dim=1).unsqueeze(-1)
        #     # assemble_features=torch.sum(assemble_features *assemble_weights,dim=1,keepdim=True)/ torch.sum(assemble_weights,dim=1,keepdim=True)
        #     return assemble_features
