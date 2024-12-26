





from tqdm import tqdm 

import numpy as np

import random
import torch
from torch.utils.data import Dataset


import random
import torch
from torch.utils.data import Dataset
from lyus.Frame import  Mapper
import os 

class Subset(torch.utils.data.Subset):

    def __getattr__(self, name):
        """
        如果尝试访问的属性在这个类中不存在，则尝试从self.dataset中调用。
        """
        return getattr(self.dataset, name)

    
    

class PartDatasetTool(object):

    def __init__(self,dataset,class_key="y"):
        self.dataset=dataset
        self.class_key=class_key
        self.class_indices=dataset.get_class_indices()


    # def preprocess_dataset(self,dataset):
    #     class_indices = {}
    #     for i, sample in tqdm(enumerate(dataset)):
    #         label = sample[self.class_key]
    #         if label not in class_indices:
    #             class_indices[label] = []
    #         class_indices[label].append(i)
    #     return class_indices
    
    def get_suport_query_data(self,k_shot, seed=None):
        import copy 
        class_indices=copy.deepcopy(self.class_indices)
    
        support_shuffled_indices = []
        query_shuffled_indices = []

        for indices in class_indices.values():
            class_shuffled_indices = indices[:]
            random.seed(seed)
            random.shuffle(class_shuffled_indices)
            support_shuffled_indices.extend(class_shuffled_indices[:k_shot])
            query_shuffled_indices.extend(class_shuffled_indices[k_shot:])
            # print(support_shuffled_indices)
        # Create a subset of the dataset using the selected indices
        support_dataset = Subset(self.dataset, support_shuffled_indices)
        query_dataset = Subset(self.dataset, query_shuffled_indices)
        return support_dataset,query_dataset
    