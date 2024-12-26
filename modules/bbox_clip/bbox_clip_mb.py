import numpy as np
from collections import Counter

from .bbox_clip import  BboxClip
from .visual import  draw_bbox_on_grid
from sklearn.model_selection import train_test_split
import  clip
import torch
from PIL import  Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import faiss

# import torch.multiprocessing as mp

# # 设置进程启动方法为 'spawn'
# mp.set_start_method('spawn')


def print_class_distribution(labels, title, previous_counts=None):
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    print(title)
    for class_label, count in sorted(class_counts.items()):
        change_str = ""
        if previous_counts is not None and class_label in previous_counts:
            change = count - previous_counts[class_label]
            change_str = f" ({'+' if change > 0 else ''}{change})"
        print(f"Class {class_label}: {count} samples{change_str}, {count / total_samples:.2%} of the dataset")
    print(f"Total: {total_samples} samples\n")
    return class_counts


def resample_data(img_embeddings, labels, indexes, max_samples_per_class):
    if not isinstance(img_embeddings, np.ndarray):
        raise TypeError("img_embeddings should be a numpy array")
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels should be a numpy array")
    if not isinstance(indexes, np.ndarray):
        raise TypeError("indexes should be a numpy array")
    if not isinstance(max_samples_per_class, int):
        raise TypeError("max_samples_per_class should be an integer")

    if img_embeddings.ndim != 2:
        raise ValueError("img_embeddings should be a 2-dimensional array")
    if labels.ndim != 1:
        raise ValueError("labels should be a 1-dimensional array")
    if indexes.ndim != 1:
        raise ValueError("indexes should be a 1-dimensional array")
    if img_embeddings.shape[0] != labels.shape[0] or img_embeddings.shape[0] != indexes.shape[0]:
        raise ValueError("The number of rows in img_embeddings, labels, and indexes should match")

    if max_samples_per_class <= 0:
        raise ValueError("max_samples_per_class should be a positive integer")

    previous_counts = print_class_distribution(labels, "Class distribution before resampling:")

    class_counts = Counter(labels)
    classes_to_resample = [class_label for class_label, count in class_counts.items() if count > max_samples_per_class]

    resampled_embeddings = []
    resampled_labels = []
    resampled_indexes = []

    for class_label in classes_to_resample:
        class_indices = np.where(labels == class_label)[0]
        _, resampled_class_indices = train_test_split(class_indices, test_size=max_samples_per_class, random_state=42)
        resampled_embeddings.append(img_embeddings[resampled_class_indices])
        resampled_labels.append(labels[resampled_class_indices])
        resampled_indexes.append(indexes[resampled_class_indices])

    classes_not_to_resample = set(labels) - set(classes_to_resample)
    for class_label in classes_not_to_resample:
        class_indices = np.where(labels == class_label)[0]
        resampled_embeddings.append(img_embeddings[class_indices])
        resampled_labels.append(labels[class_indices])
        resampled_indexes.append(indexes[class_indices])

    resampled_embeddings = np.vstack(resampled_embeddings)
    resampled_labels = np.concatenate(resampled_labels)
    resampled_indexes = np.concatenate(resampled_indexes)

    print_class_distribution(resampled_labels, "Class distribution after resampling:", previous_counts)

    return resampled_embeddings, resampled_labels, resampled_indexes


class MemoryBank(object):
    
    def __init__(self,model):
        self.bind_model(model)
        
    def bind_model(self,model):
        self.model =model
    
    
    
    def save(self, filename):
        # print(self.mb.shape, len(self.meta_data))
        data = {"img_embeddings": self.model.img_embeddings, "img_meta_data": self.model.img_meta_data,
                "text_embeddings": self.model.text_embeddings, "text_meta_data": self.model.text_meta_data
               }
        # print(self.model.text_embeddings.shape,self.model.img_embeddings.shape)
        # print(self.model.text_meta_data)
        # print(self.model.img_meta_data)
        torch.save(data, filename)


    def load(self, filename):
        data = torch.load(filename)
        self.model.img_embeddings = data["img_embeddings"] # Move the tensor back to the original device (e.g., CUDA)
        self.model.img_meta_data = data["img_meta_data"]
        self.model.text_embeddings = data["text_embeddings"] # Move the tensor back to the original device (e.g., CUDA)
        self.model.text_meta_data = data["text_meta_data"]
        
        
class BboxClipMB(object):

    def __init__(self,clip_wight):
        self.clip = BboxClip()
        self.clip.load(clip_wight)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.preprocess = self.get_clip_process()

    def get_clip_process(self):
        def _convert_image_to_rgb(image):
            return image.convert("RGB")

        try:
            from torchvision.transforms import InterpolationMode
            BICUBIC = InterpolationMode.BICUBIC
        except ImportError:
            BICUBIC = Image.BICUBIC

        def _transform(n_px):
            return Compose([
                # Resize(n_px, interpolation=BICUBIC),
                # CenterCrop(n_px),
                # _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        return _transform(None)

    def load_text_features(self, classes, texts):
        self.clip.eval()
        self.classes = classes
        text_inputs = [clip.tokenize(text) for text in texts]
        # self.classes.append("normal")
        text_inputs = torch.cat(text_inputs).to(self.device)
        with torch.no_grad():
            self.text_embeddings = self.clip.encode_text(text_inputs)
            self.text_embeddings /= self.text_embeddings.norm(dim=-1, keepdim=True)
            # self.text_embeddings=self.text_embeddings/ self.text_embeddings.norm(dim=-1, keepdim=True)
        self.text_meta_data=[ {"class":cls,"text":text} for cls,text in zip(classes,texts) ]
            
    def get_roi_embeds(self, imgList, bboxes:torch.Tensor):
        self.clip.eval()

        imgList=[  self.preprocess(img) if isinstance(img, Image.Image) else img  for img in imgList ]


        img_tensor = torch.stack([ img.to(self.device) for img in imgList])
        bboxes = bboxes.unsqueeze(0)
        pool_roi_embeds = self.clip.encode_image(img_tensor, bboxes)
        pool_roi_embeds /= pool_roi_embeds.norm(dim=-1, keepdim=True)
        return pool_roi_embeds

    def cul_similarity(self, image_features, labels=None, topK=5):
        similarity = (100.0 * image_features @ self.text_embeddings.T).softmax(dim=-1)

        # 获取最相似的前5个类别索引
        values, indices = similarity.topk(topK, dim=-1)
        if not labels: labels = ["" for i in range(image_features.shape[0])]
        results = []
        for i, (value, index, label) in enumerate(zip(values, indices, labels)):
            #  for image i
            # print(f"Image {i+1}:")
            # print(f"Actual Label: {label}")
            # print("Top 5 Predictions:")
            # for v, idx in zip(value, index):
            #     print(f"{self.classes[idx]:>16s}: {100 * v.item():.2f}%")
            # print("="*30)
            result_i = [(self.classes[idx], v.item()) for v, idx in zip(value, index)]
            results.append(result_i)
        return results

    def visual(self, imgList, bboxes):

        assert len(imgList) == len(bboxes)
        with torch.no_grad():
            img_feat = self.get_roi_embeds(imgList, bboxes)
        self.cul_similarity(img_feat)
        image, bbox = imgList[0], bboxes[0]
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Visualization
        fig, axarr = plt.subplots(1, 3, figsize=(18, 6))

        # Original image with bbox
        axarr[0].imshow(image.detach().cpu().numpy())
        rect_original = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1,
                                          edgecolor='r', facecolor='none')
        axarr[0].add_patch(rect_original)
        axarr[0].set_title("Original Image")

        draw_bbox_on_grid(axarr[1], self.clip.feat_shape, self.clip.roi_bboxes[0], self.clip.roi_list[0])
        # print(self.clip.roi_shape,self.clip.roi_bboxes[0])
        draw_bbox_on_grid(axarr[2], self.clip.roi_shape, self.clip.roi_bboxes[0])

        plt.tight_layout()
        plt.show()

    def preprocess_embeddings(self, embeddings: np.ndarray):
        embeddings = embeddings.astype("float32")
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        # faiss.normalize_L2(embeddings)
        return embeddings

    def load_img_features(self, train_dataset,batch_size=32,num_workers=4,resample=False,method="consine"):
        assert method in [ "consine" ,"l2"]
        self.clip.eval()
        img_embeddings = []
        labels = []
        from tqdm import tqdm
        from data_wrapper import DatasetWrapper
        import numpy as np
        import faiss
        
        # images_dataloader=DataLoader(images,batch_size=30)
        # DatasetWrapper(train_dataset,new_length=256)
        data_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        # train_dataset.set_clip_process()
        for idx, data in enumerate(tqdm(data_loader)):

            img_batch = data['image'].to(self.device)
            box_list = data["box_list"]
            label_batch = data["label"]
            with torch.no_grad():
                # emb=self.get_roi_embeds([i_img], i_box_list)
                pool_roi_embeds = self.clip.encode_image(img_batch, box_list)
                pool_roi_embeds /= pool_roi_embeds.norm(dim=-1, keepdim=True)
            labels.extend(label_batch)
            img_embeddings.append(pool_roi_embeds)

        img_embeddings = torch.concat(img_embeddings, dim=0).detach().cpu().numpy()
        self.img_embeddings=img_embeddings
        self.img_meta_data=train_dataset.get_img_meta_data()
        # labels=torch.concat(labels,dim=0).detach().cpu().numpy()
        # import numpy as np
        # indexes = [i for i in range(len(labels))]
        # img_embeddings,labels,indexes=img_embeddings, np.array(labels), np.array(indexes)
        # if resample:
        #     img_embeddings, labels, indexes = resample_data(img_embeddings,labels,indexes, 2000)
        # sample
        self.bulid_faiss_index()

    def search_images(self, emb, k=10):
        # text_emb=model.encode_text(clip.tokenize(text,truncate=True).to(DEVICE)).detach().cpu().numpy()
        emb = self.preprocess_embeddings(emb)
        faiss_cos_sim, k_nearest_indexes = self.faiss_index.search(emb, k=k)
        return faiss_cos_sim, k_nearest_indexes


    def save(self,mb_path):
        MemoryBank(self).save(mb_path)
        
    def load(self,mb_path):
        MemoryBank(self).load(mb_path) 
        print("building faiss index...")
        self.embed_index = list(range(len(self.img_meta_data)))
        self.bulid_faiss_index()
        
    def bulid_faiss_index(self,method="consine"):
        img_embeddings=self.img_embeddings
        # img_embeddings.shape
        faiss_embeddings = self.preprocess_embeddings(img_embeddings)
        faiss_index=None
        if method =="consine":
            faiss_index = faiss.IndexFlatIP(faiss_embeddings.shape[1])
        if method=="l2":
            faiss_index = faiss.IndexFlatL2(faiss_embeddings.shape[1])  # 使用欧式距离
        faiss_index = faiss.IndexIDMap(faiss_index)
        faiss_index.train(faiss_embeddings)
        faiss_index.add_with_ids(faiss_embeddings, np.arange(faiss_embeddings.shape[0]))
        self.faiss_index = faiss_index
  
        
