import torch
import torch.nn.functional as F
from torch import nn


def deepCopyTorchLayer(src):
    if isinstance(src, nn.Linear):
        # 创建一个新的 nn.Linear 层，具有相同的输入和输出维度
        dest = nn.Linear(src.in_features, src.out_features)
        # 复制权重和偏置
        dest.weight.data = src.weight.data.clone()
        dest.bias.data = src.bias.data.clone()
        return dest
    elif isinstance(src, nn.Parameter):
        # 创建一个新的 nn.Parameter，并复制源参数的值
        dest = nn.Parameter(src.data.clone())
        return dest
    else:
        raise ValueError(f"Unsupported layer type: {type(src)}")
        
        
        


class ClipTester:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('RN50x4', self.device)
        self.text_encoder =  lambda x: self.model.encode_text(x)


    def load_text_features(self, classes,texts):
        self.classes = classes
        texts=[clip.tokenize(text) for text in texts]
        # self.classes.append("normal")
        text_inputs = torch.cat(texts).to(self.device)
        with torch.no_grad():
            self.text_features = self.text_encoder(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            
    def predict(self, imgList, labels=None,topK=5):
        image_inputs = torch.stack([self.preprocess(img) for img in imgList]).to(self.device)
        if not labels: labels=[ "" for i in range(len(imgList)) ]
        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        
        # 获取最相似的前5个类别索引
        values, indices = similarity.topk(topK, dim=-1)
        
        for i, (value, index, label) in enumerate(zip(values, indices, labels)):
            print(f"Image {i+1}:")
            print(f"Actual Label: {label}")
            print("Top 5 Predictions:")
            for v, idx in zip(value, index):
                print(f"{self.classes[idx]:>16s}: {100 * v.item():.2f}%")
            print("="*30)