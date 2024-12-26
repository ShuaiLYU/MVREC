import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_bbox_on_grid(ax,size, bboxIn,boxOut=None):
    """
    在黑色背景上画带有红色网格线的图像，并在指定的bbox内填充颜色

    :param size: 一个元组，定义了图像的宽度和高度 (width, height)
    :param bbox: 一个元组，定义了bbox的 (x, y, width, height)
    """
    width, height = size

    

    
    # 设置坐标轴的界限
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    
    # 设置坐标轴的背景颜色
    ax.set_facecolor("black")
    
    # 画网格线
    ax.set_xticks(np.arange(width+1))
    ax.set_yticks(np.arange(height+1))
    ax.grid(which="both", linestyle='-', linewidth=1, color='red')
    
    if boxOut is None:
        boxOut=[0,0,width,height]
    
    # 在bbox内填充颜色
    bbox_x, bbox_y, bbox_x2, bbox_y2 = boxOut
    rect = patches.Rectangle((bbox_x, bbox_y), bbox_x2-bbox_x, bbox_y2-bbox_y, linewidth=0, facecolor='white', alpha=0.5)
    ax.add_patch(rect)
    

    boxinb=[  bboxIn[0]+boxOut[0],bboxIn[1]+boxOut[1],bboxIn[2]+boxOut[0],bboxIn[3]+boxOut[1] ]
    bbox_x, bbox_y, bbox_x2, bbox_y2 = boxinb
    rect = patches.Rectangle((bbox_x, bbox_y), bbox_x2-bbox_x, bbox_y2-bbox_y, linewidth=0, facecolor='orange', alpha=1)
    ax.add_patch(rect)
    
    
    
    
    
def visualize_predictions(image, boxes, label, predictions, font_size=60):
    # 在图片上画框
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline="red", width=3)

    # 查找支持中文的字体文件
    font_path="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    if font_path is None:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        # font = ImageFont.load_default()
        # print("Warning: Could not find a Chinese font. Using default font.")
    else:
        
        font = ImageFont.truetype(font_path, font_size)

    # 准备文本图像
    max_text_length = max(len(label), max(len(pred_label) for pred_label, _ in predictions))
    text_image_width = int(max_text_length * font_size * 1.2)  # Adjusted for larger font size
    text_image_height = image.height  # Fixed height for simplicity
    text_image = Image.new('RGB', (text_image_width, text_image_height), 'white')
    d = ImageDraw.Draw(text_image)
    
    d.text((10, 10), '实际标签:', fill="black", font=font)
    d.text((10, 10 + font_size), label, fill="black", font=font)
    
    d.text((10, 10 + 3 * font_size), 'Top 5 预测:', fill="black", font=font)
    for i, (pred_label, conf) in enumerate(predictions, start=1):
        d.text((10, 10 + (3 + i) * font_size), f'{pred_label}: {conf:.2%}', fill="black", font=font)
    
    # 拼接图片和文本
    final_image = Image.new('RGB', (image.width + text_image.width, image.height))
    final_image.paste(image, (0, 0))
    final_image.paste(text_image, (image.width, (image.height - text_image.height) // 2))  # Center the text image
    
    ImageDraw.Draw(final_image).rectangle([1,1,final_image.width-1,final_image.height-1], outline="blue", width=3)
    return final_image



import matplotlib.pyplot as plt
from PIL import Image

def visualize_sudoku_images(images,figsize=(20, 20)):# 九宫格可视化
    if len(images) != 9:
        raise ValueError("images列表的长度应该为9")
    
    fig, axes = plt.subplots(3, 3,figsize=figsize )
    axes = axes.flatten()
    
    for i, img in enumerate(images):
        if isinstance(img, Image.Image):
            axes[i].imshow(img)
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, "Not a PIL Image", ha='center', va='center')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

    
# import matplotlib.pyplot as plt
# from PIL import Image

# def visualize_sudoku_images(images,figsize=(20, 20)):# 九宫格可视化
#     if len(images) != 9:
#         raise ValueError("images列表的长度应该为9")
    
#     fig, axes = plt.subplots(3, 3,figsize=figsize )
#     axes = axes.flatten()
    
#     for i, img in enumerate(images):
#         if isinstance(img, Image.Image):
#             axes[i].imshow(img)
#             axes[i].axis('off')
#         else:
#             axes[i].text(0.5, 0.5, "Not a PIL Image", ha='center', va='center')
#             axes[i].axis('off')
    
#     plt.tight_layout()
#     plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_with_tsne(features, labels=None, ids=None):
    """
    Visualizes the given high-dimensional features using t-SNE with category-specific colors and shows IDs.

    Parameters:
    - features: numpy array of shape (n_samples, n_features)
    - labels: list of categories/labels for each feature vector
    - ids: list of IDs for each feature vector
    """
    # Apply t-SNE
    if ids is None:idxs = ids=list(range(features.shape[0]))  # IDs from 0 to 99
    if labels is None: labels=np.array(  [ 0  for i in range(model.text_embeddings.shape[0]) ] )
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    perplexity_value = min(30, features.shape[0] - 1)  # Adjust perplexity based on the number of samples

    # Apply t-SNE with adjusted perplexity
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    tsne_results = tsne.fit_transform(features)
    unique_labels=np.unique(labels)
    label_color_dict = dict(zip(unique_labels, colors))

    # Plot the t-SNE results with colors based on labels and add IDs
    plt.figure(figsize=(10, 10))
    for label in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == label]
        for idx in idxs:
            plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c=[label_color_dict[label]], s=50)
            plt.text(tsne_results[idx, 0], tsne_results[idx, 1], str(ids[idx]), fontsize=9)

    plt.title('t-SNE Visualization of Features with IDs')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()
    
    
    
from   torchvision import transforms
class TensorVisual(object):
    
    def __init__(self,mean,std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        self.inverseNorm=transforms.Normalize(mean=mean_inv, std=std_inv)
    
    def tensor2image(self,tensor:torch.Tensor,multiply=255,return_np=False):
        assert len(tensor.shape)==3
        from PIL import Image
        import numpy as np
        image=self.inverseNorm(tensor.float())*multiply
        # print(torch.unique(image))
        image=image.detach().cpu().numpy().transpose(1,2,0).astype(np.uint8)
        if return_np: return image
        # print(image.shape)
        return  Image.fromarray(image)
    
    
    def mb_visual(self, model, imgList, bboxes:torch.Tensor):

        assert len(imgList) == bboxes.shape[0]
        with torch.no_grad():
            img_feat = model.get_roi_embeds(imgList, bboxes)
        model.cul_similarity(img_feat)
        # print(torch.max(imgList[0]),torch.min(imgList[0]),"lyus")
        image, bboxes = imgList[0], bboxes[0]
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Visualization
        fig, axarr = plt.subplots(1, 3, figsize=(18, 6))
        image=self.tensor2image(image)
        # print(np.maximum(imgList[0]),np.minimum(imgList[0]),"lyus")
        # Original image with bbox
        axarr[0].imshow(image)
        for i in range(bboxes.shape[0]):
            bbox=bboxes[i].tolist()
            print(bbox)
            rect_original = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1,
                                              edgecolor='r', facecolor='none')
            axarr[0].add_patch(rect_original)
        axarr[0].set_title("Original Image")

        draw_bbox_on_grid(axarr[1], model.clip.feat_shape, model.clip.roi_bboxes[0], model.clip.roi_list[0])
        # print(model.clip.roi_shape,model.clip.roi_bboxes[0])
        draw_bbox_on_grid(axarr[2], model.clip.roi_shape, model.clip.roi_bboxes[0])

        plt.tight_layout()
        plt.show()

        
import pandas as pd
import numpy as np
class ConfusionMatrix(object):

    def __init__(self,classNames):

        self.labels=[]
        self.scores = []
        self.classNames=classNames
        self.numClass=len(classNames)
        self.confusionMatrix = np.zeros((self.numClass,)*2)
        print("classificationMetric, cls: {}".format(self.numClass))


    def reset(self):
        self.labels=[]
        self.scores=[]
        self.confusionMatrix=np.zeros((self.numClass, self.numClass))
        return self
    def add_batch(self,predicts,labels):
        cm=self.genConfusionMatrix(predicts,labels)
        self.confusionMatrix+=cm
        
        
#     def add_batch(self,scores,labels):
#         assert labels.shape==scores.shape[:-1]
#         self.labels.append(labels)
#         self.scores.append(scores)
#         predicts=np.argmax(scores,axis=-1)
#         cm=self.genConfusionMatrix(predicts,labels)
#         self.confusionMatrix+=cm
        
    def genConfusionMatrix(self, Predict, Label):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (Label >= 0) & (Label < self.numClass)
        label = self.numClass * Label[mask] + Predict[mask]
        print(np.unique(label))
        count = np.bincount(label.astype(np.int32), minlength=self.numClass**2)
        cm = count.reshape(self.numClass, self.numClass)
        return cm

    # def show_cm_map(self,figsize=(15,10),dpi=200):
    #     fig = plt.figure(figsize=figsize)
    #     df=pd.DataFrame(self.confusionMatrix.astype(np.int32),index=self.classNames,columns=self.classNames)
    #     fig=sns.heatmap(df,annot=True,fmt="g")
    #     plt.show()


    def show_cm_map(self, figsize=(12,6), dpi=200):
        
        # 创建索引和列的标签，格式为 'id: className'
        label_names = [f'{name} {idx}' for idx, name in enumerate( self.classNames)]
        
        # 创建一个DataFrame来存储混淆矩阵
        df = pd.DataFrame(self.confusionMatrix.astype(np.int32), index=label_names, columns=label_names)
        
        # 创建一个遮罩，只显示对角线上的值
        mask = np.zeros_like(df, dtype=bool)
        np.fill_diagonal(mask, True)
        
        # 绘制热图，对角线上的字体颜色为红色
        fig = plt.figure(figsize=figsize, dpi=dpi)
        sns.heatmap(df, annot=True, fmt="g", mask=mask, cbar=False)  # 非对角线上的值
        sns.heatmap(df, annot=True, fmt="g", mask=~mask, cbar=False, annot_kws={"color": "green"})  # 对角线上的值
        # 添加x轴和y轴的名字
        plt.xlabel('Predict')
        plt.ylabel('Label')
        plt.show()