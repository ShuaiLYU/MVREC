import torch
import torch.nn as nn

class ChannelWeights(nn.Module):
    def __init__(self, num_channels):
        super(ChannelWeights, self).__init__()
        # 初始化权重向量，权重的数量等于通道数
        # 使用 nn.Parameter 来确保这些权重是可训练的
        self.weights = nn.Parameter(torch.ones(num_channels))
        
    def forward(self, x):
        # 应用通道权重
        # 这里使用的是广播机制，确保权重向量能正确地应用到输入 x 的每个样本上
        return x * self.weights
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 使用全连接层实现注意力机制
        self.fc1 = nn.Linear(num_channels, num_channels // reduction_ratio)
        self.fc2 = nn.Linear(num_channels // reduction_ratio, num_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 两层全连接网络，中间加ReLU激活函数
        attention = self.fc1(x)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        # Sigmoid将注意力值压缩到0和1之间
        attention = self.sigmoid(attention)
        # 调整原始输入
        return x * attention+x
    


import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchTripletLoss(nn.Module):
    def __init__(self, margin=0.2, distance_type='euclidean'):
        super(BatchTripletLoss, self).__init__()
        self.margin = margin
        self.distance_type = distance_type

    def forward(self, embeddings, labels ):
    #    s= outputs["embeddings"],inputs["y"]
        # 根据距离类型计算两两距离
        if self.distance_type == 'cosine':
            # 使用余弦相似度并转换为距离
            pairwise_similarity = F.cosine_similarity(embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=2)
            pairwise_distance = 1 - pairwise_similarity
        elif self.distance_type == 'euclidean':
            pairwise_distance = torch.cdist(embeddings.float(), embeddings.float(), p=2)
        else:
            raise ValueError("Unsupported distance type. Supported types: 'cosine', 'euclidean'")

        # 创建正负样本掩码
        identity_mask = torch.eye(len(labels), device=labels.device).bool()
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~identity_mask
        negative_mask = (labels.unsqueeze(0) != labels.unsqueeze(1))
        
        # 对每个anchor, 找到最困难的正样本和负样本
        max_positive_dist = (pairwise_distance * positive_mask.float()).max(dim=1)[0]
        min_negative_dist = (pairwise_distance + ~negative_mask * 999).min(dim=1)[0]

        # 计算三元组损失
        triplet_loss = F.relu(max_positive_dist - min_negative_dist + self.margin)

        return triplet_loss.mean()
        # return {"triplet_loss":triplet_loss.mean()}
    
class SimSiamHeader(nn.Module):
    """SimSiamHeader."""
    def __init__(self, input_dim,hidden_dim,output_dim ):
        super(SimSiamHeader, self).__init__()
        self.predictor  = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
   
        b,v,c=z.shape
        assert v==2
        z1,z2= z[:,0],z[:,1]
        # Forward pass through the backbone and the heads
        p1, p2  = self.predictor(z1), self.predictor(z2)
        # Return both projections and predictions
        Loss = self.D(p1, z2) / 2 + self.D(p2, z1) / 2
        return Loss

    @staticmethod
    def D(p, z, version='simplified'): # negative cosine similarity
        if version == 'original':
            z = z.detach() # stop gradient
            p = F.normalize(p, dim=1) # l2-normalize 
            z = F.normalize(z, dim=1) # l2-normalize 
            return -(p*z).sum(dim=1).mean()

        elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
            return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
        else:
            raise Exception


def merge_first_two_dims(tensor):
    """
    Merge the first two dimensions of a given tensor.

    Args:
    tensor (torch.Tensor): The input tensor with shape (d1, d2, ..., dn).

    Returns:
    torch.Tensor: A tensor with the first two dimensions merged into one,
                  resulting in a shape of (d1*d2, ..., dn).
    """
    if tensor.dim() < 2:
        raise ValueError("Tensor must have at least two dimensions to merge the first two dimensions")
    
    # Get the original size of the tensor
    original_size = tensor.size()
    new_size = (original_size[0] * original_size[1],) + original_size[2:]

    # Reshape the tensor
    reshaped_tensor = tensor.reshape(new_size)
    return reshaped_tensor


def merge_second_third_dims(tensor):
    """

    """
    if tensor.dim() < 3:
        raise ValueError("Tensor must have at least two dimensions to merge the first two dimensions")
    
    # Get the original size of the tensor
    original_size = tensor.size()
    new_size = (original_size[0] ,original_size[1]* original_size[2] ) + original_size[3:]

    # Reshape the tensor
    reshaped_tensor = tensor.reshape(new_size)
    return reshaped_tensor


import torch
import torch.nn as nn
import torch.nn.functional as F

# Define your loss function
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        from lyus.Frame import Experiment
        self.alpha=Experiment().get_param().debug.trip_loss_weight
        self.triplet_loss = BatchTripletLoss(margin=Experiment().get_param().debug.trip_loss_margin)
        # self.alpha = alpha  # Weighting factor for the triplet loss

    def forward(self, logits, embeddings, targets):
        # Calculate the cross-entropy loss
        ce_loss = self.cross_entropy_loss(logits, targets)
        
        # Generate the triplet loss
        triplet_loss = self.triplet_loss(embeddings, targets)

        # Combine the losses
        combined_loss = ce_loss + self.alpha * triplet_loss
        return combined_loss

    # def generate_triplets(self, embeddings, targets):
    #     # This function should generate triplets (anchor, positive, negative)
    #     # For simplicity, we assume that the batch is well-formed and sorted
    #     # so that each consecutive three elements form a valid triplet.
    #     anchor = embeddings[::3]
    #     positive = embeddings[1::3]
    #     negative = embeddings[2::3]
    #     return anchor, positive, negative

    def generate_triplets(self, embeddings, targets):
        # Initialize lists to hold the triplets
        if hasattr(self, "anchor"):
            return self.anchor, self.positive, self.negative

        anchor_list = []
        positive_list = []
        negative_list = []

        # Initialize dictionaries to store positive and negative indices for each class
        if not hasattr(self, "pos_indices_dict") or not hasattr(self, "neg_indices_dict"):
            self.pos_indices_dict = {}
            self.neg_indices_dict = {}
            # Iterate over each unique class in the targets
            unique_classes = torch.unique(targets)
            for target_class in unique_classes:
                # Get the indices of all samples with the current class
                pos_indices = torch.where(targets == target_class)[0]
                self.pos_indices_dict[target_class.item()] = pos_indices

                # Get the indices of all samples with a different class
                neg_indices = torch.where(targets != target_class)[0]
                self.neg_indices_dict[target_class.item()] = neg_indices

        # Generate triplets
        for target_class, pos_indices in self.pos_indices_dict.items():
            neg_indices = self.neg_indices_dict[target_class]

            # Form anchor-positive pairs and select a random negative for each pair
            for i in range(len(pos_indices)):
                for j in range(i + 1, len(pos_indices)):
                    anchor_list.append(embeddings[pos_indices[i]])
                    positive_list.append(embeddings[pos_indices[j]])
                    negative_index = torch.randint(len(neg_indices), (1,)).item()
                    negative_list.append(embeddings[neg_indices[negative_index]])

        # Stack the lists to form tensors
        anchor = torch.stack(anchor_list)
        positive = torch.stack(positive_list)
        negative = torch.stack(negative_list)

        return anchor, positive, negative


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


import torch

def save_tensor(tensor, file_path):
    """
    Save a tensor to a local file.
    
    Args:
    tensor (torch.Tensor): The tensor you want to save.
    file_path (str): The path to the file where the tensor will be saved.
    """
    try:
        # Save the tensor to the specified file
        torch.save(tensor, file_path)
        print(f"Tensor has been successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving tensor: {e}")


import torch
import torch.nn.functional as F

def calculate_uncertainty(probs):
    """
    Calculate uncertainty based on entropy.
    
    Args:
    probs (torch.Tensor): Probabilities of shape (batch_size, num_views, num_classes)
    
    Returns:
    torch.Tensor: Uncertainty values of shape (batch_size, num_views)
    """
    probs = torch.clamp(probs, min=1e-9, max=1.0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=2)  # Calculate entropy for each view
    uncertainty = entropy / torch.log(torch.tensor(probs.size(2), dtype=torch.float))  # Normalize entropy
    return uncertainty

def weighted_logits_with_uncertainty(v):
    """
    Modify logits using uncertainty to improve performance.
    
    Args:
    v (torch.Tensor): Logits of shape (batch_size, num_views, num_classes)
    
    Returns:
    torch.Tensor: Weighted logits of shape (batch_size, num_classes)
    """
    batch_sz, num_view, num_classes = v.size()
    # print(v.size())
    
    # Calculate uncertainty
    uncertainty = calculate_uncertainty(v)  # [batch_size, num_views]
    # print(uncertainty)
    # save_tensor(uncertainty,"uncertainty.pt")
    # assert False
    # Normalize uncertainty to use as weights
    from lyus.Frame import Experiment
    alpha=Experiment().get_param().debug.uncertainty_alpha
    def activate(x):
        return x 
        # return  ((-alpha) * (1 - x)).exp()
    weights =activate( 1 - uncertainty ) # [batch_size, num_views]
    
    # print(weights[0])
    # Normalize weights so they sum to 1 for each sample
    weights = weights / weights.sum(dim=1, keepdim=True)  # [batch_size, num_views]

    # Apply weights to logits
    weighted_v = v * weights.unsqueeze(2)  # [batch_size, num_views, num_classes]
    
    # Calculate weighted mean
    weighted_mean_v = weighted_v.sum(dim=1,keepdim=False)  # [batch_size, num_classes]
    
    return weighted_mean_v.squeeze()
    # assert False
    # Calculate probabilities from logits
    probs = F.softmax(v, dim=-1)  # [batch_size, num_views, num_classes]



class AdapterModule(nn.Module):
    def __init__(self, c_in, reduction=None,use_zi=True):
        super(AdapterModule, self).__init__()


        self.adapter = nn.Sequential(
            # nn.LayerNorm(c_in),
            nn.Linear(c_in, c_in , bias=True), # 
            # nn.ReLU(inplace=True),
            nn.SiLU(inplace=True),
            # nn.Linear(c_in // reduction, c_in, bias=True),
            # nn.SiLU(inplace=True)
        )
        if use_zi:
            self.init_weights()

    def init_weights(self):
        pass
        for m in self.adapter:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        # nn.init.zeros_(self.adapter[2].weight)
        # self.adapter[2]
    def forward(self, x):
        # print(self.adapter[0].weight)
        return self.adapter(x)+x