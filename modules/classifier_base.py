import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn.functional as F
from .torch_demo import CombinedLoss,merge_second_third_dims,merge_first_two_dims
from .torch_demo import merge_first_two_dims

import os 

def cosine_similarity_between_x_and_x0(x):
    # Extract y from x
    y = x[:, :, 0, :]  # Shape [a, b, d]

    # Normalize x and y along the last dimension
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)

    # Compute the cosine similarity using batch matrix multiplication
    cosine_similarity = torch.einsum('abcd,abd->abc', x_norm, y_norm)

    return cosine_similarity

def compute_class_prototypes(class_prototype, class_indexs, num_classes=None):
    """
    计算每个类别的原型向量

    参数:
    - class_prototype (torch.Tensor): [N*K, 678] 的矩阵，表示每个样本的特征向量
    - class_indexs (torch.Tensor): [N*K, 1] 的向量，表示每个样本的类别索引
    - num_classes (int): 类别的总数 K

    返回:
    - prototypes (torch.Tensor): [K, 678] 的矩阵，表示每个类别的原型向量
    """
    # 创建存储每个类别向量的容器
    if num_classes==None:
        num_classes=torch.max(class_indexs)+1
    prototypes = torch.zeros(num_classes, class_prototype.size(1)).to(class_prototype.device)  # 存储每个类的 prototype
    counts = torch.zeros(num_classes, 1).to(class_prototype.device)   # 统计每个类的向量个数

    # 计算每个类别的原型向量
    for i in range(class_prototype.size(0)):
        class_idx = class_indexs[i].item()  # 获取当前样本的类别索引
        prototypes[class_idx] += class_prototype[i]  # 累加属于该类的向量
        counts[class_idx] += 1  # 该类向量个数加 1

    # 计算平均值（避免除以 0）
    prototypes = prototypes / counts # clamp(min=1) 避免除以 0
    # print(prototypes)    
    # print(counts)
    # assert False

    return prototypes



class BaseClassifier(nn.Module):

    def __init__(self, *args, **kwargs):
        super(BaseClassifier,self).__init__()



    def forward_zs_clip(self, image_features):
        x = image_features

        # image_features = self.ratio * x + (1 - self.ratio) * image_features
        image_features=x
        text_features =self.text_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        embeddings=image_features
        return logits



    def get_loss(self):
        def ClassLoss(inputs, outputs):
            ce_loss = F.cross_entropy(outputs["logits"], inputs["y"])
            return {"ce_loss": ce_loss }
    def init_weight(self, cache_keys:torch.tensor, cache_vals:torch.tensor,fineturn=True):
        pass

    def save_model(self, data_name: str, feature_name: str, few_shot: str, sampling_id: str, dirname: str):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = f"{data_name}_{feature_name}_{few_shot}_{sampling_id}.pt"
        filepath = os.path.join(dirname, filename)
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def check_and_load_model_weight(self, data_name: str, feature_name: str, few_shot: str, sampling_id: str, dirname: str) -> bool:
        filename = f"{data_name}_{feature_name}_{few_shot}_{sampling_id}.pt"
        filepath = os.path.join(dirname, filename)
        if os.path.exists(filepath):
            try:
                state_dict = torch.load(filepath)
                model_state_dict = self.state_dict()
                missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())
                unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())
                
                self.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"Missing keys when loading state_dict: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys when loading state_dict: {unexpected_keys}")
                
                print(f"Model weights loaded from {filepath}")
                return True
            except Exception as e:
                print(f"Error loading model weights from {filepath}: {e}")
                return False
        else:
            print(f"No model weights found at {filepath}")
            return False
        
    def train_compact(self, features, one_hot_labels):
        assert len(features.shape)==3 and len(one_hot_labels.shape)==2
        one_hot_labels=one_hot_labels.unsqueeze(1).repeat(1,features.shape[1],1)
        from .torch_demo import merge_first_two_dims
        one_hot_labels=merge_first_two_dims(one_hot_labels)
        features=merge_first_two_dims(features)

        labels = one_hot_labels.argmax(dim=1)
           # 训练步骤数量
        from lyus.Frame import Experiment
        total_steps =int(Experiment().get_param().debug.ft_epo) 
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

                pbar.set_postfix(loss=loss.item(), lr=current_lr, tau=f"{self.tau:.3f}")
                pbar.update()



        # 切换到评估模式
        self.eval()

    def set_trainable_params(self, trainable_settings):
        # trainable_settings=["backbone.pool2D.q_proj"]
        
       
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


    def train_compact_with_embed(self, features, one_hot_labels):


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
        # criterion = nn.CrossEntropyLoss()
        criterion = CombinedLoss()      
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
                    loss = criterion(outputs["logits"],outputs["embeddings"], targets)
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
                
                pbar.set_postfix(loss=loss.item(), lr=current_lr, alpha=f"{self.alpha:.4f}", tau=f"{self.tau:.3f}")
                pbar.update()



        # 切换到评估模式
        self.eval()
