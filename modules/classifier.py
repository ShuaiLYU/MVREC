import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn.functional as F
from .classifier_base import *
from .torch_demo import CombinedLoss,merge_second_third_dims,merge_first_two_dims
from .torch_demo import merge_first_two_dims, AdapterModule
    #  from .torch_demo import merge_second_third_dims
from lyus.Frame import Experiment

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision.models import resnet18






class EuclideanClassifier(BaseClassifier):

    def forward(self, x):

        # Compute the squared Euclidean distance
        # Using broadcasting to expand A_norm and B_norm for vectorized subtraction
        dist_squared = torch.sum((x.unsqueeze(1) - self.proto.unsqueeze(0)) ** 2, dim=2)
        
        # Convert squared distances to distances
        distances = torch.sqrt(dist_squared + 1e-9)  # Add small epsilon for numerical stability
        
        # Convert distances to similarities (negative scaling by tau)
        # The smaller the distance, the higher the resulting similarity
        similarities = -self.tau * distances
        
        # Compute the softmax over the negative distances to obtain probabilities
        predicts = F.softmax(similarities, dim=1)

        return {"predicts": predicts, "similarities": similarities}
    

    def init_weight(self, cache_keys: torch.Tensor, cache_vals: torch.Tensor):
        assert len(cache_keys.shape) == 4
        cache_keys = merge_second_third_dims(cache_keys)

        assert len(cache_keys.shape) == 3

        assert len(cache_vals.shape) == 2
        nk_class_index = cache_vals.argmax(dim=-1)
        assert cache_keys.shape[0] == cache_vals.shape[0], f"{cache_keys.shape} and {cache_vals.shape}"

        k_img_prototype= compute_class_prototypes(cache_keys.mean(dim=1,keepdim=False),nk_class_index)
        self.proto=k_img_prototype
        self.tau=0.11





class CosimClassfier(EuclideanClassifier):

    def forward(self, x):

        # Normalize the input tensors
        A_norm = F.normalize(x, p=2, dim=1)
        B_norm = F.normalize(self.proto, p=2, dim=1)
     
        # Compute the cosine similarity
        cosim = torch.mm(A_norm, B_norm.t())
    
        # Ensure self.tau is not zero to avoid division by zero
        # You might also want to ensure self.tau is not too small to avoid numerical instability
        tau = self.tau if self.tau and self.tau > 1e-9 else 1e-9
        
        # Add a small epsilon to the denominator to prevent division by zero
        epsilon = 1e-9
        logits = cosim / (tau + epsilon)
        # print(logits)
        predicts = logits.softmax(dim=-1)
        # print(predicts)

        return { "predicts": predicts, "logits": logits}
    

class KNNClassifier(BaseClassifier):
    def __init__(self):
        super(KNNClassifier, self).__init__()
        self.k = 1
        self.tau = 0.11
        self.epsilon = 1e-9  # Small epsilon for numerical stability

    def forward(self, x):
        # Compute cosine similarity
        # Normalize both x and self.proto
        x_normalized = F.normalize(x, p=2, dim=1)  # Normalize input x
        proto_normalized = F.normalize(self.proto, p=2, dim=1)  # Normalize prototypes

        # Compute cosine similarity matrix
        similarities = torch.matmul(x_normalized, proto_normalized.t())  # [batch_size, num_prototypes]

        # Find the k-nearest neighbors
        topk_similarities, indices = similarities.topk(self.k, largest=True, dim=1)  # Use largest=True for cosine similarity

        # Gather the one-hot labels of the k-nearest neighbors
        knn_labels = self.one_hot_labels[indices]  # [batch_size, k, num_classes]

        # Compute the logits by summing the weighted one-hot labels
        logits = torch.sum(self.tau * topk_similarities.unsqueeze(2) * knn_labels, dim=1)  # [batch_size, num_classes]

        # Compute the softmax over the logits to obtain probabilities
        predicts = F.softmax(logits, dim=1)

        return {"predicts": predicts, "logits": logits}


    def init_weight(self, cache_keys: torch.Tensor, cache_vals: torch.Tensor):
        assert len(cache_keys.shape) == 4
        cache_keys = merge_second_third_dims(cache_keys) # n*K, v*l, c 
        assert len(cache_keys.shape) == 3
        nk_img_prototype=cache_keys.mean(dim=1, keepdim=False)
        assert len(cache_vals.shape) == 2

        self.proto = nk_img_prototype
        self.one_hot_labels = cache_vals  # Labels corresponding to the prototypes
        self.num_classes = cache_vals.shape[1]
        self.tau = 0.11
        self.k= (cache_keys.shape[0]) // self.num_classes
        print(f"set KNN 's K to {self.k}")


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np


class LinearProbeClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(max_iter=500, solver='lbfgs')
        super(LinearProbeClassifier, self).__init__()
    def forward(self, x):  # x.shape (batch, feature_len)
        device = x.device
        x = x.cpu().numpy()
        logits = self.predict(x)
        predicts = logits
        logits = torch.tensor(logits, device=device)
        predicts = torch.tensor(predicts, device=device)
        return {"predicts": predicts, "logits": logits}

    def fit(self, features, labels):
        # Normalize the features
        features = self.scaler.fit_transform(features)
        
        # Train the logistic regression classifier
        self.classifier.fit(features, labels)

    def predict(self, features):
        features = self.scaler.transform(features)
        logits = self.classifier.decision_function(features)  # returns (batch, num_classes)
        return logits

    def init_weight(self, cache_keys: torch.Tensor, cache_vals: torch.Tensor):
        assert len(cache_keys.shape) == 4
        cache_keys = merge_second_third_dims(cache_keys)
        assert len(cache_keys.shape) == 3
        assert len(cache_vals.shape) == 2
        assert cache_keys.shape[0] == cache_vals.shape[0], f"{cache_keys.shape} and {cache_vals.shape}"

        nk_class_index = cache_vals.argmax(dim=-1)

        nk_img_prototype = cache_keys.mean(dim=1, keepdim=False).cpu().numpy()

        # nk_img_prototype=merge_first_two_dims(cache_keys).cpu().numpy()
        # nk_class_index=nk_class_index.unsqueeze(1).repeat(1,cache_keys.shape[1],1)
        # nk_class_index=merge_first_two_dims(nk_class_index).cpu().numpy()


        self.fit(nk_img_prototype, nk_class_index.cpu().numpy())
  


class ClipZeroShot(BaseClassifier):
    def __init__(self,text_feature,logit_scale, tau=0.11):
        super(ClipZeroShot, self).__init__()
        self.tau = tau
        assert tau is not None
        self.text_feature=text_feature
        assert len(text_feature.shape )==2
        c_in= text_feature.shape[1]
        self.logit_scale=logit_scale
        self.adapter=nn.Identity()
        
    def init_weight(self, cache_keys: torch.Tensor, cache_vals: torch.Tensor):
        pass

    def _forward(self, image_features):
        x = self.adapter(image_features)

        # image_features = self.ratio * x + (1 - self.ratio) * image_features
        image_features=x
        text_features =self.text_feature

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        embeddings=image_features
        return logits,embeddings


    def forward(self, x):

        logits,embeddings= self._forward(x)
        predicts = logits.softmax(dim=-1)
        return { "predicts": predicts, "logits": logits,"embeddings":embeddings}


class ClipAdapter(BaseClassifier):

    def __init__(self,text_feature,logit_scale, tau=0.11):
        super(ClipAdapter, self).__init__()
        self.tau = tau
        assert tau is not None
        self.text_feature=text_feature
        assert len(text_feature.shape )==2
        c_in= text_feature.shape[1]
        self.logit_scale=logit_scale
        reduction=4   #  https://github.com/gaopengcuhk/CLIP-Adapter/blob/main/clip_adapter.py
        self.ratio=0.2 #  https://github.com/gaopengcuhk/CLIP-Adapter/blob/main/clip_adapter.py

        self.adapter = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
    


    def _forward(self, image_features):
        x = self.adapter(image_features)

        # image_features = self.ratio * x + (1 - self.ratio) * image_features
        image_features=x
        text_features =self.text_feature

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        embeddings=image_features
        return logits,embeddings


    def forward(self, x):

        logits,embeddings= self._forward(x)
        predicts = logits.softmax(dim=-1)
        return { "predicts": predicts, "logits": logits,"embeddings":embeddings}
    
    def set_trainable_params(self):
        # trainable_settings=["backbone.pool2D.q_proj"]
        
        trainable_settings=["adapter"]
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
        
    def init_weight(self, cache_keys: torch.Tensor, cache_vals: torch.Tensor):
        assert len(cache_keys.shape) == 4
        cache_keys = merge_second_third_dims(cache_keys)

        assert len(cache_keys.shape) == 3
        assert len(cache_vals.shape) == 2
        assert cache_keys.shape[0] == cache_vals.shape[0], f"{cache_keys.shape} and {cache_vals.shape}"

        nk_class_index = cache_vals.argmax(dim=-1).cpu().numpy()

        self.set_trainable_params()
        # self.train_compact(cache_keys, cache_vals)

        from lyus.Frame import Experiment
        k_shot=Experiment().get_param().debug.k_shot
        feat_name=Experiment().get_param().ClipModel.clip_name
        data_name=Experiment().get_param().data_option
        sampling_id=Experiment().get("sampling_id")

      
        self.train_compact(cache_keys, cache_vals)
        return 
        # self.save_model(data_name=data_name,feature_name=feat_name,few_shot=k_shot,sampling_id=sampling_id,dirname="./classifier")
        # if self.check_and_load_model_weight(data_name=data_name,feature_name=feat_name,few_shot=k_shot,sampling_id=sampling_id,dirname="./classifier"):
        #     pass
        # else:
        #     self.train_compact(cache_keys, cache_vals)
        #     self.save_model(data_name=data_name,feature_name=feat_name,few_shot=k_shot,sampling_id=sampling_id,dirname="./classifier")



class ClassificationAdapter(ClipAdapter):

    def __init__(self,text_feature):
        super(ClipAdapter, self).__init__()
        self.tau = 0.11
        self.alpha=0
        # assert tau is not None
        self.text_feature=text_feature
        assert len(text_feature.shape )==2
        c_in= text_feature.shape[1]
        # self.logit_scale=logit_scale
        reduction=4   #  https://github.com/gaopengcuhk/CLIP-Adapter/blob/main/clip_adapter.py
        self.ratio=0.2 #  https://github.com/gaopengcuhk/CLIP-Adapter/blob/main/clip_adapter.py

        # self.adapter = nn.Sequential(
        #     nn.Linear(c_in, c_in // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(c_in // reduction, c_in, bias=False),
        #     nn.ReLU(inplace=True)
        # )
        self.adapter= AdapterModule(c_in,reduction )
        # self.adapter=nn.Identity()
        num_classes= text_feature.shape[0]
        self.classifer_head=nn.Linear(c_in, num_classes)



    def _forward(self, image_features):
        x = self.adapter(image_features)
        # image_features = self.ratio * x + (1 - self.ratio) * image_features
        image_features=x
        logits = self.classifer_head(image_features)
        # print(logits.shape)
        # assert False
        return logits,image_features


    def set_trainable_params(self):
        # trainable_settings=["backbone.pool2D.q_proj"]
        
        trainable_settings=["adapter","classifer_head"]
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
    
        
    def init_weight(self, cache_keys: torch.Tensor, cache_vals: torch.Tensor):
        assert len(cache_keys.shape) == 4
        cache_keys = merge_second_third_dims(cache_keys)

        assert len(cache_keys.shape) == 3
        assert len(cache_vals.shape) == 2
        assert cache_keys.shape[0] == cache_vals.shape[0], f"{cache_keys.shape} and {cache_vals.shape}"

        nk_class_index = cache_vals.argmax(dim=-1).cpu().numpy()

        self.set_trainable_params()

        self.train_compact(cache_keys, cache_vals)
        # self.train_compact_with_embed(cache_keys, cache_vals)
        return 




class ProxyNet(BaseClassifier):
    def __init__(self,proto, tau=0.11):
        super(CosimClassfier, self).__init__()
        self.tau = tau
        assert tau is not None
        self.proto=proto
        # self.channel_attention = ChannelAttention(num_channels)
        # self.channel_attention = ChannelWeights(num_channels)
    def forward(self, x):

        # print(self.channel_attention.weights)
        # x = self.channel_attention(x)
        # proto = self.channel_attention(proto)
        # Normalize the input tensors
        A_norm = F.normalize(x, p=2, dim=1)
        B_norm = F.normalize(self.proto, p=2, dim=1)
     
        # Compute the cosine similarity
        cosim = torch.mm(A_norm, B_norm.t())
    
        # Ensure self.tau is not zero to avoid division by zero
        # You might also want to ensure self.tau is not too small to avoid numerical instability
        tau = self.tau if self.tau and self.tau > 1e-9 else 1e-9
        
        # Add a small epsilon to the denominator to prevent division by zero
        epsilon = 1e-9
        logits = cosim / (tau + epsilon)
        # print(logits)
        predicts = logits.softmax(dim=-1)
        # print(predicts)

        return { "predicts": predicts, "logits": logits}




class TipAdapter(BaseClassifier):
    def __init__(self,text_features, tau=0.11):
        super(TipAdapter, self).__init__()
        # self.tau = tau
        assert tau is not None
        # self.channel_attention = ChannelAttention(num_channels)
        # self.channel_attention = ChannelWeights(num_channels)
        labels= list(range(text_features.shape[0]))
        weight_y=[]
        for class_label in labels:
            y= torch.tensor([class_label])
            y_one_hot= torch.nn.functional.one_hot(y,len(labels))
            weight_y.append(y_one_hot) 
        self.weight_y=torch.concat(weight_y).to(text_features.device)
        self.register_buffer("one_hot_class",self.weight_y.float())

    def forward(self, x):
        # alpha=20



        alpha =int(Experiment().get_param().debug.acti_beta) 


        x = F.normalize(x, p=2, dim=1)
        x = x @ self.dense1_weight.t()
        x= ((-1) * (alpha - alpha * x)).exp()
        # x=F.relu(x)
        logits = x @ self.dense2_weight

        tau = self.tau if self.tau and self.tau > 1e-9 else 1e-9
        logits = logits / (tau + 1e-9)
        predicts = logits.softmax(dim=-1)

        return { "predicts": predicts, "logits": logits}


    def get_loss(self):
        def ClassLoss(inputs, outputs):
            ce_loss = F.cross_entropy(outputs["logits"], inputs["y"])
            return {"ce_loss": ce_loss }


        return ClassLoss
    def init_weight(self, cache_keys:torch.tensor, cache_vals:torch.tensor):


        assert len(cache_keys.shape) ==4
        ck4d= cache_keys
        # cache_keys=merge_second_third_dims(cache_keys)
        cache_keys=cache_keys[:,:,0,:]

        assert len(cache_keys.shape) ==3
        assert len(cache_vals.shape) ==2
        assert cache_keys.shape[0] ==cache_vals.shape[0], f" {cache_keys.shape} and  {cache_vals.shape}"

        nk_class_index=cache_vals.argmax(dim=-1)



        nk_img_prototype= cache_keys.mean(dim=1,keepdim=False)
        nk_img_prototype_norm=F.normalize(nk_img_prototype.float(), p=2, dim=1)
        k_img_prototype= compute_class_prototypes(cache_keys.mean(dim=1,keepdim=False),nk_class_index)
        k_img_prototype_norm=F.normalize(k_img_prototype.float(), p=2, dim=1)
     

        self.dense1_weight = nn.Parameter(nk_img_prototype_norm,requires_grad=True)  # Initialize alpha as a learnable parameter
        self.register_buffer("dense2_weight",cache_vals.float())
        # self.register_buffer("alpha",torch.tensor(20.0) )
        self.register_buffer("tau",torch.tensor(0.11) )


    


class TipAdapterF(TipAdapter):

    def init_weight(self, cache_keys:torch.tensor, cache_vals:torch.tensor):
        
        TipAdapter.init_weight(self,cache_keys=cache_keys,cache_vals=cache_vals)

        assert len(cache_keys.shape) ==4

        cache_keys3d=merge_second_third_dims(cache_keys) # b,v*l,c 

        self.train_compact(cache_keys3d,cache_vals)



class SdpaModule(nn.Module):


    def __init__(self, support_key,support_class, class_proxies,scale):
        super(SdpaModule, self).__init__()
        self.class_proxies=class_proxies
        from lyus.Frame import Experiment
        self.support_class=support_class  # [ class index,... ]  shape N*K
        # self.class_proxies=class_proxies
        self.class_proxies=class_proxies
        # scale=10
        self.scale=scale

    
    def forward (self,query,support_key):
        # Step 1: Compute the dot product between query and key
        # attn_logits = torch.matmul(query, self.support_key.transpose(-2, -1))
        key=support_key

        query=F.normalize(query, p=2, dim=1)
        key=F.normalize(key, p=2, dim=1)

        attn_logits = torch.matmul(query, key.transpose(-2, -1))

        # Step 2: Scale the dot product
        attn_logits = attn_logits 
        alpha=self.scale

        # Step 3: Apply softmax to get attention weights
        attn_weights= ((-1) * (alpha - alpha * attn_logits)).exp()
        # attn_weights = F.softmax(attn_logits* self.scale*10, dim=-1)   
        suport_values=  self.class_proxies[self.support_class] 
        # print(query.shape,attn_weights.shape)
        # assert False
        logits=torch.matmul(attn_weights, suport_values)@self.class_proxies.t() 

        return logits


# class ClipAdapterPlus(ClipAdapter):
#     def __init__(self,text_feature,logit_scale, tau=0.11):
#         super(ClipAdapterPlus, self).__init__(text_feature,logit_scale,tau)
    

#     def forward(self, x):

#         logits= self.clip_forward(x)
#         predicts = logits.softmax(dim=-1)
#         return { "predicts": predicts, "logits": logits}
    



from .torch_demo import Projector


class EchoClassfier(TipAdapter):

    def __init__(self,text_features, tau=0.11,**kwargs):
        super(EchoClassfier, self).__init__(text_features,tau)
        # self.tau = tau
        assert tau is not None
        # self.channel_attention = ChannelAttention(num_channels)
        # self.channel_attention = ChannelWeights(num_channels)
        labels= list(range(text_features.shape[0]))
        weight_y=[]
        for class_label in labels:
            y= torch.tensor([class_label])
            y_one_hot= torch.nn.functional.one_hot(y,len(labels))
            weight_y.append(y_one_hot) 
        self.weight_y=torch.concat(weight_y).to(text_features.device)
        self.register_buffer("one_hot_class",self.weight_y.float())
        self.logit_scale= kwargs.get("logit_scale",None)
        weight_x=text_features.unsqueeze(1)
        text_features=F.normalize(text_features, p=2, dim=1) 
        # self.init_weight(weight_x,weight_y,False)
  
        self.text_features_tensor=text_features
        self.zifa= AdapterModule(text_features.shape[1])
    def forward(self, x):
        # x = F.normalize(x, p=2, dim=1) # temp change

        embeddings= self.zifa(x)
        support_key= self.zifa(self.support_key)
        # logits,embeddings = torch.stack([ sdpa(x) for sdpa in  self.msdpa]).mean(dim=0,keepdim=False)
        logits_list = []
        for sdpa in self.msdpa:
            logits = sdpa(embeddings,support_key)  # Unpack the tuple
            logits_list.append(logits)
        # Stack and mean
        logits = torch.stack(logits_list).mean(dim=0, keepdim=False)
        tau = self.tau if self.tau and self.tau > 1e-9 else 1e-9
        logits = logits / (tau + 1e-9)
        predicts = logits.softmax(dim=-1)
        # predicts= logits
        return { "predicts": predicts, "logits": logits,"embeddings":embeddings}     



    def init_weight(self, cache_keys:torch.tensor, cache_vals:torch.tensor):
        self._init_weight(cache_keys,cache_vals,"onehot")


    def _init_weight(self, cache_keys:torch.tensor, cache_vals:torch.tensor,proxy_style):
        assert len(cache_keys.shape) ==4
        ck4d= cache_keys
        cache_keys=merge_second_third_dims(cache_keys)
        # cache_keys=cache_keys[:,:,0,:]
        assert len(cache_vals.shape) ==2


   
     
        # res=cosine_similarity_between_x_and_x0(ck4d)
        # res=res.view(res.shape[0],-1).mean(dim=1,keepdim=False)
        # print(res)

        assert cache_keys.shape[0] ==cache_vals.shape[0], f" {cache_keys.shape} and  {cache_vals.shape}"
        # print(cache_keys.shape)

        nk_class_index=cache_vals.argmax(dim=-1)
        self.nk_class_index=nk_class_index
        nk_img_prototype= cache_keys.mean(dim=1,keepdim=False)
        nk_img_prototype_norm=F.normalize(nk_img_prototype.float(), p=2, dim=1)
        k_img_prototype= compute_class_prototypes(cache_keys.mean(dim=1,keepdim=False),nk_class_index)
        k_img_prototype_norm=F.normalize(k_img_prototype.float(), p=2, dim=1)
    
        dense2_weight=cache_vals.float()


        self.register_buffer("alpha",torch.tensor(20.0) )
        # self.tau = nn.Parameter(torch.tensor(0.33),requires_grad=True)  # Initialize alpha as a learnable parameter
        self.register_buffer("tau",torch.tensor(0.11) )

        from lyus.Frame import  Experiment
        sdpa_scale=Experiment().get_param().debug.sdpa_scale
        # proxy= self.text_features_tensor
        # proxy=

        proxies=[self.weight_y.float()]
        # proxies=[self.weight_y.float(), self.text_features_tensor]
        self.register_buffer("k_img_prototype",k_img_prototype )
        # proxies=[self.text_features_tensor.float()]
        # scale_list=[1000]




     

        from lyus.Frame import Experiment
        acti_beta =int(Experiment().get_param().debug.acti_beta) 


        assert proxy_style in ["onehot","text","onehot_text"]
        if  proxy_style=="onehot":
            scale_list=[acti_beta]
            proxies=[self.weight_y.float()]
        elif proxy_style=="text":
            scale_list=[acti_beta]
            proxies=[self.text_features_tensor.float()]
        elif proxy_style=="onehot_text":
            scale_list=[1,48]
            # assert False
            proxies=[self.weight_y.float(),self.text_features_tensor.float()]


        self.support_key= nn.Parameter(nk_img_prototype)  

        self.msdpa=torch.nn.Sequential(*[SdpaModule(None,nk_class_index,proxy,scale=scale)
                    for proxy,scale in zip(proxies,scale_list)])



 

    def train_compact(self, features, one_hot_labels,use_triple_loss=False):


        assert len(features.shape)==3 and len(one_hot_labels.shape)==2,features.shape
        one_hot_labels=one_hot_labels.unsqueeze(1).repeat(1,features.shape[1],1)
        from .torch_demo import merge_first_two_dims
        one_hot_labels=merge_first_two_dims(one_hot_labels)
        features=merge_first_two_dims(features)

        labels = one_hot_labels.argmax(dim=1)

        # # 检查features数量是否小于32，如果是则复制扩展
        # if len(features) < 32:
        #     multiplier = 32 // len(features) + 1
        #     features = features.repeat(multiplier, 1)
        #     labels = labels.repeat(multiplier)

           # 训练步骤数量
        from lyus.Frame import Experiment
        total_steps =int(Experiment().get_param().debug.ft_epo) 
        print_step=20
        batch_size=len(features)
        shuffle=True
        lr= 1e-4 #1e-4 
        # print(dict(self.named_parameters()))
        # assert False
        optimizer = optim.AdamW(self.parameters(), lr=1e-4,eps=1e-6)

        # optimizer = optim.AdamW([
        #     {'params': [self.tau], 'lr': lr *1},  # alpha with 100 times the base learning rate
        #     {'params': [param for name, param in self.named_parameters() if name != 'tau'], 'lr': lr}
        # ])

        # optimizer = optim.SGD(self.parameters(), lr=1e-3)
        # optimizer = optim.Adam(self.parameters(), lr=1e-4,eps=1e-6)
        # 创建数据加载器
        dataset = TensorDataset(features, labels)


        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # 定义损失函数和优化器
  
        if use_triple_loss:
            criterion = CombinedLoss()      
        else:
            # assert False
            criterion = nn.CrossEntropyLoss() 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps,eta_min=lr*0.1)
        # 设定训练模式
        self.train()

        # Creating a generator to fetch data from dataloader
        def batch_generator(dataloader):
            while True:  # Infinite loop to continuously provide data
                for data in dataloader:
                    yield data

        # Instantiate the generator
        gen = batch_generator(loader)

        # Training loop with progress bar
        with tqdm(total=total_steps, desc="Training Progress") as pbar:
            for step in range(total_steps):
                inputs, targets = next(gen)  # Get batch from generator

                with torch.cuda.amp.autocast():
                    # Forward pass
                    outputs = self(inputs)
                    if use_triple_loss:
                        loss = criterion(outputs["logits"],outputs["embeddings"], targets)
                    else:
                         loss = criterion(outputs["logits"], targets)
                        # Early stopping if loss falls below 0.05
                # if loss.item() < 0.02:
                #     print("Early stopping triggered as loss fell below 0.05")
                #     break
                # Backward and optimize
                optimizer.zero_grad()
                # loss.backward()
                # print(loss)
                loss.backward(retain_graph=True)
                #Print layers with gradients
                if not hasattr(self,"print_grad_event"):
                    self.print_grad_event=True
                    print("Layers with gradients:")
                    for name, param in self.named_parameters():
                        if param.grad is not None:
                            print(f"{name} has gradient")
                        else:
                            print(f"{name} does not have gradient")
                                # torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                optimizer.step()
                scheduler.step()
                
                # Fetch current learning rate from the scheduler
                current_lr = scheduler.get_last_lr()[0]  # Assumes scheduler provides a list of learning rates

                # Update the progress bar with loss information

                pbar.set_postfix(loss=loss.item(), lr=current_lr, alpha=f"{self.alpha:.4f}", tau=f"{self.tau.item():.3f}")
                pbar.update()



        # 切换到评估模式
        self.eval()

 
class EchoClassfierT(EchoClassfier):



    def init_weight(self, cache_keys:torch.tensor, cache_vals:torch.tensor):
        
        EchoClassfier.init_weight(self,cache_keys=cache_keys,cache_vals=cache_vals)

        assert len(cache_keys.shape) ==4

        self.set_trainable_params(["zifa"])
        cache_keys3d=merge_second_third_dims(cache_keys) # b,v*l,c 
        self.train_compact(cache_keys3d,cache_vals)



# class EchoClassfierF(EchoClassfier):

#     def init_weight(self, cache_keys:torch.tensor, cache_vals:torch.tensor):
        
#         EchoClassfier.init_weight(self,cache_keys=cache_keys,cache_vals=cache_vals)

#         assert len(cache_keys.shape) ==4

#         self.set_trainable_params(["support_key","zifa"])
#         cache_keys3d=merge_second_third_dims(cache_keys) # b,v*l,c 
#         self.train_compact(cache_keys3d,cache_vals)


class EchoClassfierF(EchoClassfier):


    def __init__(self,text_features, tau=0.11,**kwargs):
        super(EchoClassfier, self).__init__(text_features,tau)

        from lyus.Frame import Experiment
        zip_config_index=int(Experiment().get_param().debug.zip_config_index)
        configs= [
               {"trainable_params":["zifa"],"use_zi":False,"use_triple_loss":False },  # all false 0

            {"trainable_params":["support_key","zifa"],"use_zi":False,"use_triple_loss":False },  #1
            {"trainable_params":["zifa"],"use_zi":True,"use_triple_loss":False },#2
            {"trainable_params":["zifa"],"use_zi":False,"use_triple_loss":True }, #3

            {"trainable_params":["zifa"],"use_zi":True,"use_triple_loss":True },#4
            {"trainable_params":["support_key","zifa"],"use_zi":False,"use_triple_loss":True }, #5
            {"trainable_params":["support_key","zifa"],"use_zi":True,"use_triple_loss":False },  #6
      
            {"trainable_params":["support_key","zifa"],"use_zi":True,"use_triple_loss":True },# ￥7
            {"trainable_params":["support_key"],"use_zi":True,"use_triple_loss":False },  # 8
            # {"trainable_params":["support_key","zifa"],"use_zi":True,"use_triple_loss":False },#9

        ]

        self.config= configs[zip_config_index]

        # self.tau = tau
        assert tau is not None
        # self.channel_attention = ChannelAttention(num_channels)
        # self.channel_attention = ChannelWeights(num_channels)
        labels= list(range(text_features.shape[0]))
        weight_y=[]
        for class_label in labels:
            y= torch.tensor([class_label])
            y_one_hot= torch.nn.functional.one_hot(y,len(labels))
            weight_y.append(y_one_hot) 
        self.weight_y=torch.concat(weight_y).to(text_features.device)
        self.register_buffer("one_hot_class",self.weight_y.float())
        self.logit_scale= kwargs.get("logit_scale",None)
        weight_x=text_features.unsqueeze(1)
        text_features=F.normalize(text_features, p=2, dim=1) 
        # self.init_weight(weight_x,weight_y,False)
  
        self.text_features_tensor=text_features
        self.zifa= AdapterModule(text_features.shape[1],use_zi=self.config["use_zi"])

    def save_feature_xy(self,feature, ys, name):
        assert len(feature.shape) == 2 
        # assert len(ys.shape) == 2 

        # Convert tensors to numpy arrays
        feature = feature.detach().cpu().numpy()
        ys = ys.detach().cpu().numpy()
        # support_key
        # Define file names
        feature_file = f"{name}_features.npy"
        ys_file = f"{name}_labels.npy"
        dirname = os.path.basename(Experiment().get_save_dir())
        savedir="/home/lyushuai/Projects/MVREC_notebook/save" +f"/{dirname}"
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        # Save the numpy arrays to disk
        np.save(savedir+"/"+feature_file, feature)
        np.save(savedir+"/"+ys_file, ys)

        print(f"Features saved to {feature_file}")
        print(f"Labels saved to {ys_file}")


    def init_weight(self, cache_keys:torch.tensor, cache_vals:torch.tensor):
        
        EchoClassfier.init_weight(self,cache_keys=cache_keys,cache_vals=cache_vals)

        self.save_feature_xy(self.support_key,self.nk_class_index,"before_train")
        assert len(cache_keys.shape) ==4
        self.set_trainable_params(self.config["trainable_params"])
        cache_keys3d=merge_second_third_dims(cache_keys) # b,v*l,c 
        self.train_compact(cache_keys3d,cache_vals,use_triple_loss=self.config["use_triple_loss"])

    
        self.save_feature_xy(self.support_key,self.nk_class_index,"after_train")
        # k_img_prototype weight_y




class EchoClassfier_text(EchoClassfier):

    def forward(self, x):
        # x = F.normalize(x, p=2, dim=1) # temp change

        embeddings= self.zifa(x)
        support_key= self.zifa(self.support_key)
        # logits,embeddings = torch.stack([ sdpa(x) for sdpa in  self.msdpa]).mean(dim=0,keepdim=False)
        logits_list = []
        for sdpa in self.msdpa:
            logits = sdpa(embeddings,support_key)  # Unpack the tuple
            logits_list.append(logits)
        # Stack and mean
        logits = torch.stack(logits_list).mean(dim=0, keepdim=False)
        tau = self.tau if self.tau and self.tau > 1e-9 else 1e-9
        logits = logits / (tau + 1e-9)

        text_logits= self.forward_zs_clip(x)
        
        text_logits_wight=Experiment().get_param().debug.text_logits_wight

        if text_logits_wight>0:
            predicts=  logits.softmax(dim=-1) *(1-text_logits_wight)+text_logits.softmax(dim=-1)*(text_logits_wight)
        else:
            predicts = logits.softmax(dim=-1) 
        # predicts= logits
        # predicts= logits
        return { "predicts": predicts, "logits": logits,"embeddings":embeddings}    

    
class EchoClassfierF_text(EchoClassfierF):

    def forward(self, x):
        # x = F.normalize(x, p=2, dim=1) # temp change

        embeddings= self.zifa(x)
        support_key= self.zifa(self.support_key)
        # logits,embeddings = torch.stack([ sdpa(x) for sdpa in  self.msdpa]).mean(dim=0,keepdim=False)
        logits_list = []
        for sdpa in self.msdpa:
            logits = sdpa(embeddings,support_key)  # Unpack the tuple
            logits_list.append(logits)
        # Stack and mean
        logits = torch.stack(logits_list).mean(dim=0, keepdim=False)
        tau = self.tau if self.tau and self.tau > 1e-9 else 1e-9
        logits = logits / (tau + 1e-9)



        text_logits= self.forward_zs_clip(x)
        
        text_logits_wight=Experiment().get_param().debug.text_logits_wight

        if text_logits_wight>0:
            predicts=  logits.softmax(dim=-1) *(1-text_logits_wight)+text_logits.softmax(dim=-1)*(text_logits_wight)
        else:
            predicts = logits.softmax(dim=-1) 
        # predicts= logits
        return { "predicts": predicts, "logits": logits,"embeddings":embeddings}    


    
# class EchoClassfierF_MP(EchoClassfier):

#     def init_weight(self, cache_keys:torch.tensor, cache_vals:torch.tensor):
        
#         EchoClassfier._init_weight(self,cache_keys=cache_keys,
#                                   cache_vals=cache_vals,
#                                   proxy_style="text")

#         assert len(cache_keys.shape) ==4

#         self.set_trainable_params(["support_key","zifa"])
#         cache_keys3d=merge_second_third_dims(cache_keys) # b,v*l,c 
#         self.train_compact(cache_keys3d,cache_vals)



 
# class SimAtten(nn.Module):

#     def __init__(self,k,v,activate):
#         super(SimAtten,self).__init__()
#         self.k=k
#         self.v=v
#         self.activate=activate
#         c_in=k.shape[-1]
#         reduction=4
#         self.adapter = nn.Sequential(
#             nn.Linear(c_in, c_in // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(c_in // reduction, c_in, bias=False),
#             nn.ReLU(inplace=True)
#         )
#         # self.adapter = nn.Identity()
#         pass

#     def forward(self,q):
#         return self.sdpa_func(q,self.k,self.v,self.adapter,self.activate)
#     def sdpa_func( self,q , k ,v,adpater, activate):
#         # q,v= adpater(q),adpater(k)
#         q=adpater(q)*0.2 +q*(1-0.2)
#         # k=adpater(k)*0.2 +k*(1-0.2)
#         q=F.normalize(q, p=2, dim=1)
#         k=F.normalize(k, p=2, dim=1)

#         return activate(q@k.t()) @ v
     

# class EchoClassfierF(EchoClassfier):
    
#     def init_weight(self, cache_keys:torch.tensor, cache_vals:torch.tensor,fineturn=True):
#         assert len(cache_keys.shape) ==4
#         ck4d= cache_keys
#         print(cache_keys.shape)
#         cache_keys=merge_second_third_dims(cache_keys)
#         # cache_keys=cache_keys[:,:,0,:]
#         assert len(cache_vals.shape) ==2



#         assert cache_keys.shape[0] ==cache_vals.shape[0], f" {cache_keys.shape} and  {cache_vals.shape}"
#         # print(cache_keys.shape)

#         nk_class_index=cache_vals.argmax(dim=-1)
#         self.nk_class_index=nk_class_index
#         nk_img_prototype= cache_keys.mean(dim=1,keepdim=False)

#         self.nk_img_prototype=nk_img_prototype
#         nk_img_prototype_norm=F.normalize(nk_img_prototype.float(), p=2, dim=1)
#         k_img_prototype= compute_class_prototypes(cache_keys.mean(dim=1,keepdim=False),nk_class_index)
#         k_img_prototype_norm=F.normalize(k_img_prototype.float(), p=2, dim=1)
 

   




#         # self.register_buffer("dense1_weight",class_prototype )
#         # self.alpha = nn.Parameter(torch.tensor(20.0),requires_grad=False)  # Initialize alpha as a learnable parameter
#         self.register_buffer("alpha",torch.tensor(20.0) )
#         # self.tau = nn.Parameter(torch.tensor(0.33),requires_grad=True)  # Initialize alpha as a learnable parameter
#         self.register_buffer("tau",torch.tensor(0.11) )

#         from lyus.Frame import Experiment
#         self.sdpa_scale=Experiment().get_param().debug.sdpa_scale
#         self.nk_class_proxies=self.text_features_tensor.float()
#         self.k_class_proxies= self.nk_class_proxies[nk_class_index]

#         proxies=[self.weight_y.float()]
#         # proxies=[self.weight_y.float(), self.text_features_tensor]
#         self.register_buffer("k_img_prototype",k_img_prototype )
#         proxies=[self.text_features_tensor.float()]

#         # proxies=[self.weight_y.float()]
#         # scale_list=[20]

#         # proxies=[self.weight_y.float(),self.text_features_tensor.float()]
#         # scale_list=[20,300]
#         c_in=nk_img_prototype.shape[-1]
#         reduction=4
#         self.adapter = nn.Sequential(
#             nn.Linear(c_in, c_in // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(c_in // reduction, c_in, bias=False),
#             nn.ReLU(inplace=True)
#         ).to(cache_keys.device)


#         # from lyus.Frame import Experiment
#         # k_shot=Experiment().get_param().debug.k_shot
#         # feat_name=Experiment().get_param().ClipModel.clip_name
#         # data_name=Experiment().get_param().data_option
#         # sampling_id=Experiment().get("sampling_id")
#         # if self.check_and_load_model_weight(data_name=data_name,feature_name=feat_name,few_shot=k_shot,sampling_id=sampling_id,dirname="./classifier"):
#         #     pass
#         # else:
#         #     assert False
#         #     # self.train_compact(cache_keys, cache_vals)
#         #     # self.save_model(data_name=data_name,feature_name=feat_name,few_shot=k_shot,sampling_id=sampling_id,dirname="./classifier")

#         self.dense2_weight=cache_vals.float()
#         # with torch.cuda.amp.autocast():
#         #     x=self.adapter(self.nk_img_prototype)*0.2 +self.nk_img_prototype*(1-0.2)
#         #     self.dense2_weight= self.clip_forward0(x)

#         self.simatten=SimAtten(k=self.text_features_tensor[self.nk_class_index],v=self.dense2_weight,
#                                activate= lambda x: x*self.logit_scale.exp().float() ).to(self.nk_img_prototype.device)
        

#         self.simatten1=SimAtten(k=self.nk_img_prototype,v=self.dense2_weight,
#                         activate= lambda x: ((-1) * (self.sdpa_scale - self.sdpa_scale * x)).exp() ).to(self.nk_img_prototype.device)

#         BaseClassifier.train_compact(self,cache_keys,cache_vals)
        

        
#     def forward(self,x):
#         # x=self.adapter(x)*0.2 +x*(1-0.2)
#         # logits= (self.cache_forward(x)+ self.clip_forward0(x))/2
#         logits= self.simatten(x)
#         # logits= (self.simatten(x)+ self.simatten1(x))/2
#         predicts = logits.softmax(dim=-1)
#         return { "predicts": predicts, "logits": logits}
#         # return self.train_forward(x)
    



# class TransformerClassifier(BaseClassifier):


    # def __init__(self,text_feature, tau=0.11):
    #     BaseClassifier.__init__(self)
    #     self.tau = tau
    #     # assert tau is not None
    #     self.text_feature=text_feature
    #     assert len(text_feature.shape )==2
    #     c_in= text_feature.shape[1]
    #     # self.logit_scale=logit_scale


    #     # 定义分类头


    # def init_weight(self, cache_keys: torch.Tensor, cache_vals: torch.Tensor):
    #     assert len(cache_keys.shape) == 4
    #     cache_keys = merge_second_third_dims(cache_keys)  #  n*k,v*l,c
    #     assert len(cache_keys.shape) == 3
    #     assert len(cache_vals.shape) == 2
    #     assert cache_keys.shape[0] == cache_vals.shape[0], f"{cache_keys.shape} and {cache_vals.shape}"

    #     nk_class_index=cache_vals.argmax(dim=-1)

    #     nk_img_prototype= cache_keys.mean(dim=1,keepdim=False)
    #     self.nk_img_prototype=nk_img_prototype

    #     # k_img_prototype= compute_class_prototypes(cache_keys.mean(dim=1,keepdim=False),nk_class_index)


    #     nk_class_index = cache_vals.argmax(dim=-1).cpu().numpy()
    #     num_classes = self.text_feature.shape[0]

    #     feature_dim = 768
 
    #     support_image_features=nk_img_prototype
    #     support_text_features=self.text_feature[nk_class_index]
    #     print(support_image_features.shape,support_text_features.shape )
    #     # 将支持集的图像特征和文本特征合并
    #     # support_features = torch.cat((support_image_features, support_text_features), dim=0)  # [num_support, feature_dim]
    #     # 设置k和v为模型的固定属性 
    #     # self.k = support_features.unsqueeze(1)  # [num_support, 1, feature_dim]
    #     # self.v = self.k  # 对于自注意力，值和键可以相同

    #     # 设置k和v为模型的固定属性
    #     self.k = support_image_features  # [num_support_images, 1, feature_dim]
    #     self.v = support_text_features  # [num_support_texts, 1, feature_dim]
    #     num_layers=1
    #     num_heads=8
    #     # 定义Transformer层
    #     self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads).to(cache_keys.device)
    #     encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads).to(cache_keys.device)
    #     self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(cache_keys.device)
    #     self.classifier = nn.Linear(feature_dim, num_classes).to(cache_keys.device)
    #     self.set_trainable_params(["transformer_encoder","classifier","multihead_attn"])
    #     # self.train_compact(cache_keys, cache_vals)

    #     from lyus.Frame import Experiment
    #     k_shot=Experiment().get_param().debug.k_shot
    #     feat_name=Experiment().get_param().ClipModel.clip_name
    #     data_name=Experiment().get_param().data_option
    #     sampling_id=Experiment().get("sampling_id")
    #     self.train_compact(cache_keys,cache_vals)
     

    
    # def tran_forward(self, q):
    #     q= q.unsqueeze(0) # [seq_len, batch_size, feature_dim]
    #     # Transformer expects input shape: [seq_len, batch_size, feature_dim]
    #     k = self.k.unsqueeze(1).repeat(1,q.size(1), 1)  # [batch_size, num_support_images, feature_dim]
    #     v = self.v.unsqueeze(1).repeat(1,q.size(1),1)  # [batch_size, num_support_texts, feature_dim]
        
    #     # qkv = torch.cat((q, k.to(q.device), v.to(q.device)), dim=1)  # [seq_len, batch_size, feature_dim]
    #     # qkv = qkv.permute(1, 0, 2)  # [seq_len, batch_size, feature_dim]
    #     # print(q.shape,k.shape,v.shape)
    #     # assert False
    #     # # Transformer编码
    #     attn_output, _= self.multihead_attn(q,k,v)
    #     print(q.shape,k.shape, v.shape ,attn_output.shape)
    #     assert False
    #     # encoded_features = self.transformer_encoder(q,k,v,key_padding_mask=None)  # [seq_len, batch_size, feature_dim]
    #     encoded_features = self.transformer_encoder(attn_output)  # [135, 1, 768]
    #     # 取出编码后的查询特征部分进行分类
    #     encoded_query_features = encoded_features[:, :, :].mean(0)  # [batch_size, feature_dim]
    #     # encoded_query_features = encoded_features.permute(1, 0, 2).mean(dim=1)
    #     # 分类
    #     logits = self.classifier(encoded_query_features)  # [batch_size, num_classes]
        
    #     return logits

    # def forward(self, x):

    #     logits= self.tran_forward(x)
    #     predicts = logits.softmax(dim=-1)
    #     return { "predicts": predicts, "logits": logits}
    