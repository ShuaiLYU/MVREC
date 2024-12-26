import time 
import lyus.Frame as FM

import torch.nn.functional as F
# 导入PyTorch相关的库
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader
from lyus.Frame import Experiment
# 定义深度学习模型的父类
class ModelBase(nn.Module):
    # 初始化模型
    def __init__(self,model_name, backbone, neck, head):
        super(ModelBase, self).__init__()
        self.backbone = backbone # backbone模块，负责提取特征
        self.neck = neck # neck模块，负责转换特征
        self.head = head # head模块，负责输出预测
        self.loss = None # loss_func模块，负责计算损失
        self.model_name=model_name
    def load_loss(self,loss):
        self.loss=loss
        self.set_mode("train")
    # 根据mode参数决定是否冻结网络权重的函数
    def set_mode(self, mode):
        # 如果mode是"train"，则解冻所有的权重，否则冻结所有的权重
        self.mode=mode
        if mode == "train":
            self.train()
            for param in self.parameters():
                param.requires_grad = True
        elif mode=="finetune":
            self.train()
            for param in self.parameters():
                param.requires_grad = True

            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    # 前向传播
    def forward(self, input_dict):
        # 从输入字典中获取输入数据和标注
        x = input_dict["x"]
        # 输入数据经过backbone，变为Embeddings
        embeddings = self.backbone(x)
        # Embeddings经过neck，变为logits
        logits = self.neck(embeddings)
        # logits经过head，变为predicts
        results = self.head(logits)
        if self.mode in ["train","valid","test","finetune"]:
            y = input_dict["y"]
            losses = self.loss(input_dict, results)
            results["losses"]=losses
        if self.mode in ["train","finetune"]:
            # print(losses)
            batch_loss_sum = sum(losses.values())
            # print(batch_loss_sum.dtype)
            batch_loss_sum.backward()
        return results






from modules.bbox_clip.bbox_clip import  BboxClip,Projector




import torch
import torch.nn as nn




from .torch_demo import merge_first_two_dims,merge_second_third_dims

# Merging the


        #    b,v,c=x.shape 
        #     z= self.predictor(x.view(-1,c))
        #     return z.view(b,v,-1)



# from .classifier import EchoClassfier,CosimClassfier,TipEcho,TipEchoT,EchoClassfierT,LinearProbeClassifier,ClipAdapter,EchoClassfierF
from .classifier import *

from .bbox_clip.bbox_clip_vit import BboxClipVit
from .bbox_clip.alpha_clip_backbone import AlphaClip, AlphaClipWomask
from .bbox_clip.vanilla_clip import VanillaClip


class ClipModel(ModelBase):

    def __init__(self,clip_name, backbone_name,classifier, input_shape,text_list ) -> None:
        self.clip_name=clip_name
        assert clip_name in ["BboxClip","BboxClipVit", "AlphaClip","VanillaClip","AlphaClipWomask"]
        DEVICE="cuda:0" if torch.cuda.is_available() else "cpu"
        model_name="BboxClipModel"

        if clip_name=="BboxClip":
            backbone=BboxClip(input_shape,"RN50x4")  #RN50x16
        elif clip_name=="AlphaClipWomask":
            backbone=AlphaClipWomask(input_shape,backbone=backbone_name)# "ViT-L/14")
        elif clip_name=="AlphaClip":
            backbone=AlphaClip(input_shape,backbone=backbone_name)# "ViT-L/14")
        elif clip_name=="VanillaClip":
            backbone=VanillaClip(input_shape,backbone=backbone_name)#"ViT-L/14")
        elif clip_name=="BboxClipVit":
            backbone=BboxClipVit(input_shape, backbone=backbone_name)# 'ViT-B/16')  #RN50
        
        # print(text_features)
        # print(text_features.shape)
        # neck=Adapter(640,320)
        text_list=backbone.tokenize(text_list).to(DEVICE)
        text_features=backbone.encode_text(text_list)
        # neck=Projector(768,320,768)
        neck=nn.Identity()

        head=None
        super(ClipModel, self).__init__(model_name,backbone,neck,head)

        self.text_features=text_features
        self.backbone=backbone
        self.classifier=classifier
        self.init_classifier()
        
        # self.ssl_head=SimSiamHeader(768,320,768)

        # print(text_features.shape,111)
        # self.set_prototype(text_features)
        # self.set_text_prototype(text_features)
    
        # print(self.proto.shape)
        # self.load_loss(BatchTripletLoss(1,"euclidean"))


    def init_classifier(self):
        classifier=self.classifier
        text_features=self.text_features
        backbone=self.backbone
        device=backbone.device
        assert classifier in  [ "ClipZeroShot",
                                "CosimClassfier",
                               "EchoClassfier" ,"EchoClassfierT","EchoClassfierF","EchoClassfier_text","EchoClassfierF_text",
                               "TipAdapter","TipAdapterF",
                               "LinearProbeClassifier",
                               "TransformerClassifier",
                               "ClipAdapter",
                               "EuclideanClassifier",
                               "KNNClassifier",
                               "ClassificationAdapter",
                               ]
        if classifier=="CosimClassfier":
            head=CosimClassfier()
        elif classifier=="EuclideanClassifier":
            head=EuclideanClassifier()
        elif classifier=="EchoClassfierT":
            head=EchoClassfierT(text_features)
        elif classifier=="EchoClassfier":
            head=EchoClassfier(text_features)
        elif classifier=="EchoClassfierF":
            head=EchoClassfierF(text_features)
        elif classifier=="EchoClassfierF_text":
            head=EchoClassfierF_text(text_features)
        elif classifier=="EchoClassfier_text":
            head=EchoClassfier_text(text_features)
        elif classifier=="EchoClassfierF_MP":
            head=EchoClassfierF_MP(text_features)
        elif classifier=="TipAdapter":
            head=TipAdapter(text_features)
        elif classifier=="TipAdapterF":
            head=TipAdapterF(text_features)
        elif classifier=="LinearProbeClassifier":
            head=LinearProbeClassifier()
        elif classifier=="ClipZeroShot":
            head=ClipZeroShot(text_features,logit_scale=backbone.copy_logit_scale_module())
        elif classifier=="ClipAdapter":
            head=ClipAdapter(text_features,logit_scale=backbone.copy_logit_scale_module())
        elif classifier=="KNNClassifier":
            head=KNNClassifier()
        elif classifier=="ClassificationAdapter":
            head=ClassificationAdapter(text_features)
        else:
            raise Exception(f"unknown head name {classifier}")

        self.head=head.to(device)
        self.head.logit_scale=backbone.copy_logit_scale_module()
        self.head.text_features=text_features
        self.load_loss(head.get_loss())


    def bbox2mask(self, bboxes,input_size=None):
        assert len(bboxes.shape) == 3  # [b, 1, 4]
        assert bboxes.shape[1] == 1
        assert bboxes.shape[2] == 4
        batch = len(bboxes)
        alpha_batch = []
        if input_size is None:  input_size = self.input_size
        for i in range(batch):
            bbox = bboxes[i, 0]  # Fix index to use i instead of 0
         
            # Initialize the binary_mask with zeros
            binary_mask = np.zeros((input_size, input_size), dtype=np.uint8)
            
            # Fill the area inside the bbox with ones
            # bbox is expected to be [x1, y1, x2, y2]
            # x1, y1 is the top-left corner and x2, y2 is the bottom-right corner of the rectangle
            x1, y1, x2, y2 = bbox
            binary_mask[y1:y2, x1:x2] = 1
            
            # Apply transformation and move to GPU
            alpha = self.mask_transform((binary_mask * 255).astype(np.uint8))
            alpha = alpha.to(torch.device('cuda')).half()  # .unsqueeze(dim=0)
            alpha_batch.append(alpha)

        alpha_batch = torch.stack(alpha_batch)
        return alpha_batch

    def set_prototype(self,proto):

        self.register_buffer("proto",proto )

    def set_text_prototype(self,text_prototypes):
        self.text_prototypes = text_prototypes
        return text_prototypes



    def get_mvrec(self,input_dict,feature_view_num=None,mulit_view=True):
        """
            return : mvrec  tensor: [batch, view_num* token_num, ch]
        """

        
        # bboxes = input_dict["bboxes"]
        # masks= self.bbox2mask(bboxes)
 
        if "mvrec" not in input_dict.keys():
            assert "masks" in input_dict.keys()
            masks=input_dict["masks"]
            image_tensor = input_dict["x"]
    
            y = input_dict["y"]
            b,multiView,c,w,h=image_tensor.shape
            assert image_tensor.shape[1]==masks.shape[1], f"{image_tensor.shape} , {masks.shape}"
            # print(image_tensor.shape)

            image_tensor=merge_first_two_dims(image_tensor) # b*multiView,c,w,h
            masks=merge_first_two_dims(masks)     
            with torch.cuda.amp.autocast():
                start_time = time.perf_counter()
                embeddings = self.backbone.encode_image( image_tensor, masks)  # b*v,l,c
                # elapsed_time = (time.perf_counter() - start_time) * 1000  # ms
                # print(f"Batch size: {image_tensor.shape[0]}, Inference time: {elapsed_time:.3f} ms")

                # from lyus.Frame import Experiment
                # mv_name=Experiment().get_param().data.mv_method
                # # Log to the specified file
                # with open("time_log1.txt", "a") as file:
           
                #     file.write(f"mv-{mv_name}.batch{image_tensor.shape[0]} executed in {elapsed_time:.3f} ms\n")
            
                _,l,e_c=embeddings.shape
                mvrec = self.neck(embeddings).view(b,-1,l,e_c)
                input_dict["mvrec"]=mvrec
    
        # assert len(input_dict["mvrec"].shape)==3 and input_dict["mvrec"].shape[1]>=837, input_dict["mvrec"].shape
        # assert False
        from lyus.Frame import Experiment
        if feature_view_num is not None:
            if mulit_view:
                input_dict["mvrec"]=input_dict["mvrec"][:,:,:int(feature_view_num),:]
            else:
                input_dict["mvrec"]=input_dict["mvrec"][:,:1,:int(feature_view_num),:]
        else:
            if mulit_view:
                input_dict["mvrec"]=input_dict["mvrec"][:,:,:,:]
            else:
                input_dict["mvrec"]=input_dict["mvrec"][:,:1,:,:]
            
        if False: # temp change
            mvrec= input_dict["mvrec"]
            mvrec=mvrec.view(mvrec.shape[0],27,-1,*mvrec.shape[2:])
            mvrec=mvrec[:,:,0,:]
            input_dict["mvrec"]=mvrec
            # print(input_dict["mvrec"].shape)  #  b, v,l, c
            # assert False
        return input_dict["mvrec"]

    def set_img_prototype(self, k_shot, dataset):

        self.init_classifier() 
        # return 
        self.set_mode("eval")  # Assuming this is a necessary custom function for setting the mode.
        # # Create a DataLoader from the dataset
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,collate_fn=dataset.get_collate_fn())

        # Initialize a dictionary to store sum of embeddings and count for each class
        class_embeddings = {}
        class_counts = {}
        with torch.no_grad():  # Disable gradient computation
            for input_dict in dataloader:

                from lyus.Frame import Experiment
                feature_view_num=Experiment().get_param().debug.fvns

                embeddings=self.get_mvrec(input_dict,feature_view_num)  # b, new_view,L,ch

                # embeddings=merge_second_third_dims(embeddings) # b, new_view*L,ch
                # embeddings=merge_first_two_dims(embeddings.transpose(1,2))
                
                # assert len(embeddings.shape)==2 or len(embeddings.shape)==3
                # if len(embeddings.shape)==3:
                #     embeddings=torch.mean(embeddings,dim=1,keepdim=False)
                y=input_dict["y"]
                # y=y.repeat_interleave(len(embeddings)//len(y))
 
                for emb, label in zip(embeddings, y):  # batch iter
                    label = label.item()
                    # if label not in class_counts or class_counts[label] < k_shot:
                    if label not in class_embeddings.keys():
                            class_embeddings[label] =[]
                            class_counts[label] = 0

                    class_embeddings[label].append(emb.clone())
                    class_counts[label] += 1

        print(class_counts)
        # Compute the average embedding for each class
        weight_x,weight_y=[],[]
        for class_label in class_embeddings:
            for embeddings in class_embeddings[class_label]:
                # embeddings=embeddings.mean(dim=0,keepdim=True)
                weight_x.append(embeddings)
                # y= torch.tensor([class_label]*embeddings.shape[0]).to(embeddings.device)
                y= torch.tensor([class_label]).to(embeddings.device)
                y_one_hot= torch.nn.functional.one_hot(y,len(class_embeddings))

                weight_y.append(y_one_hot) 
        weight_x=torch.stack(weight_x)  # shot, num_token, ch
        weight_y=torch.concat(weight_y)

        # print(weight_x.shape)
        # assert False
        self.head.init_weight(weight_x,weight_y) #

        self.proto= (weight_x,weight_y)
            

    # 前向传播
    # def forward_ssl(self,input_dict):
    #     # self.print_trainable_parameters()
    #     result_batch={}
    #     with torch.cuda.amp.autocast():
    #         embeddings=self.get_mvrec(input_dict) # b, v, c 
    #         loss= self.ssl_head(embeddings)
    #     result_batch["losses"]= {"ssl_loss":loss}
    #     return result_batch
    

    def forward(self, input_dict,mulit_view):
        

        # if "mvrec" not in input_dict.keys():
        #     # self.print_trainable_parameters()
        #     if self.mode=="ssl":
        #         return self.forward_ssl(input_dict)
        from lyus.Frame import Experiment
        feature_view_num=Experiment().get_param().debug.fvnq

        mvrec=self.get_mvrec(input_dict,feature_view_num,mulit_view)
        assert len(mvrec.shape)==4 # b, new_view,L,ch
        mvrec=merge_second_third_dims(mvrec) # b, new_view*L,ch
        infer_styles= [ "assemble_on_embed","assemble_on_logits","assemble_uncertainty"]
                
        infer_style=Experiment().get_param().debug.infer_style
        assert infer_style in infer_styles, infer_style

        if infer_style=="assemble_on_embed":
            embeddings=torch.mean(mvrec,dim=1,keepdim=False)
            with torch.cuda.amp.autocast():
                results = self.head(embeddings)
                results["embeddings"]=embeddings
        if infer_style=="assemble_on_logits":
            with torch.cuda.amp.autocast():
                src_shape=mvrec.shape[:-1]
                batch_sz=mvrec.shape[0]
                embeddings=mvrec.view(-1, mvrec.shape[-1])
                # if embeddings.shape[0]== batch_sz:
                
                results = self.head(embeddings)

                results["logits"]=results["logits"].view(batch_sz,-1,results["logits"].shape[-1])
                
                results["logits"]=results["logits"].mean(dim=1,keepdim=False)
                results["predicts"]  = results["logits"].softmax(dim=-1)
           
                # for k, v in results.items():
                #     print(k,v.shape)
        if infer_style=="assemble_uncertainty":

            with torch.cuda.amp.autocast():
                src_shape=mvrec.shape[:-1]
                batch_sz=mvrec.shape[0]
                embeddings=mvrec.view(-1, mvrec.shape[-1])
                # print(src_shape,embeddings.shape)
                results = self.head(embeddings)
                from .torch_demo import weighted_logits_with_uncertainty

                results["predicts"]=results["predicts"].view(batch_sz,-1,results["predicts"].shape[-1])
                # assert results["predicts"].shape[1]>1, results["predicts"].shape
                results["predicts"]=weighted_logits_with_uncertainty(results["predicts"]).squeeze()
                    
                results["predicts"]  = results["predicts"].softmax(dim=-1)
                   
        if self.mode in ["train","valid","test","finetune"]:
            y = input_dict["y"]
            losses = self.loss(input_dict, results)
            results["losses"]=losses
        if self.mode in ["train","finetune"]:
            # print(losses)
            batch_loss_sum = sum(losses.values())
            batch_loss_sum.backward()
            # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
        assemble_with_clip_logits=False
        if assemble_with_clip_logits:
            embeddings=torch.mean(mvrec,dim=1,keepdim=False)
            clip_logits=self.head.forward_zs_clip(embeddings)
            # print(clip_logits.shape,results["predicts"].shape)
            results["predicts"]= results["predicts"]*0.95+clip_logits.softmax(dim=-1)*0.05

        return results



    def train(self,active):
        
        super(ClipModel,self).train()
    def eval(self,active):
        super(ClipModel,self).eval()

    def set_mode(self, mode):
        # 如果mode是"train"，则解冻所有的权重，否则冻结所有的权重
        assert mode in ["train", "eval","test","infer","ssl"]
        self.mode=mode
        if mode == "train":
            self.train(False)  # fix norm layer 
            self.set_trainable_attention_params()
            
        elif mode=="ssl":
            self.eval(True)  # fix norm layer 
            self.set_trainable_ssl_params()

            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            self.eval(True)
            for param in self.parameters():
                param.requires_grad = False


    def set_trainable_attention_params(self):
        # trainable_settings=["backbone.pool2D.q_proj"]
        
        trainable_settings=["backbone.pool2D.q_proj",
                            "backbone.pool2D.k_proj",
                            "backbone.pool2D.v_proj",
                            "backbone.pool2D.c_proj",
                            "backbone.pool2D.positional_embedding"]
        # trainable_settings=["head.channel_attention"]
        # trainable_settings=["neck"]
        # Iterate over all parameters and their names
        for name, param in self.named_parameters():
            # Check if the current parameter name is in the settings dictionary

            if any([  item in name   for item in trainable_settings   ]):
               
                # Set requires_grad according to the dictionary
                param.requires_grad = True
            else:

                param.requires_grad = False  # or continue, depending on the use case
            # print(name,False)
        

    def set_trainable_ssl_params(self):
        trainable_settings=["ssl_head","neck"]
        # Iterate over all parameters and their names
        for name, param in self.named_parameters():
            # Check if the current parameter name is in the settings dictionary
            if any([  item in name   for item in trainable_settings   ]):
                param.requires_grad = True
            else:

                param.requires_grad = False  # or continue, depending on the use case

           

    def print_trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Trainable parameters: {name}, shape: {param.shape}")
