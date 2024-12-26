

from lyus.Frame.eval_tool import EvalTool
from tqdm import tqdm 
import numpy as np
import random
import torch
from torch.utils.data import Dataset, Subset
from lyus.Frame import  Mapper
import os 
from modules.part_data import PartDatasetTool

from lyus.Frame import Experiment



def gen_few_shot_support_query_dataset(dataset, k_shot, num_sampling=10):
    support_datasets = []
    query_datasets=[]

   # Preprocess the dataset to get the indices of samples for each class

    # # Get the indices of samples for each class
    # class_indices = {label: [i for i, sam in enumerate(dataset) if sam["y"] == label] for label in class_labels}
    if k_shot==0:
        print("k_shot:", 0)
        return [None]*num_sampling, [dataset]*num_sampling
    else:
        data_tool = PartDatasetTool(dataset)
        for index in tqdm(range(num_sampling)):
            support_dataset,query_dataset=data_tool.get_suport_query_data(k_shot=k_shot,seed=index)
            support_datasets.append(support_dataset)
            query_datasets.append(query_dataset)
        return support_datasets,query_datasets

# def calculate_mean_and_std(results):
#     # Initialize dictionaries to store metric values
#     metric_values = {key: [] for key in results[0].keys()}  # Assuming results is not empty

#     # Iterate over the results and collect metric values
#     for result in results:
#         for key, value in result.items():
#             metric_values[key].append(value)

#     # Calculate mean and standard deviation for each metric
#     mean_std_dict = {}
#     for key, values in metric_values.items():
#         mean_std_dict[key] = {'mean': np.mean(values), 'std': np.std(values),"num":len(results),"values":values}

#     return mean_std_dict


def calculate_mean_and_std(results):
    # Initialize dictionaries to store metric values
    metric_values = {key: [] for key in results[0].keys()}  # Assuming results is not empty

    # Iterate over the results and collect metric values
    for result in results:
        for key, value in result.items():
            metric_values[key].append(value)

    # Calculate mean and standard deviation for each metric
    mean_std_dict = {}
    for key, values in metric_values.items():
        mean = round(np.mean(values), 4)
        std = round(np.std(values), 4)
        mean_std_dict[key] = {'mean': mean, 'std': std, 'num': len(results), 'values': values}

    return mean_std_dict



def assemble_by_averge(embeddings, group_size):
    """
    Reshape the embeddings tensor from [b, ...] to [b//group_size, group_size, ...] and average across the new second dimension.

    Args:
    embeddings (torch.Tensor): The input tensor with shape [b, ...].
    group_size (int): The size of the group over which to average.

    Returns:
    torch.Tensor: The reshaped and averaged tensor with shape [b//group_size, ...].
    """
    # 确保 b 是 group_size 的倍数
    assert embeddings.size(0) % group_size == 0, f"The size of the first dimension must be a multiple of {group_size}."

    # 获取原始维度
    original_dims = embeddings.shape[1:]

    # 重塑张量到 [b//group_size, group_size, ...]
    new_shape = (-1, group_size) + original_dims
    reshaped_embeddings = embeddings.view(new_shape)

    # 沿着新的第二维度求平均
    assemble_embeddings = torch.mean(reshaped_embeddings.float(), dim=1)
    return assemble_embeddings


def merge_dicts_with_prefixes(input_dict):
    combined_res = {}

    for prefix, sub_dict in input_dict.items():
        for key, value in sub_dict.items():
            combined_res[f'{prefix}_{key}'] = value

    return combined_res

import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        # for k, v in subset[0].items(): print(k, v.shape)  # mvrec torch.Size([837, 768])
        # assert False

    def __getitem__(self, index):
        sam = self.subset[index]
        return sam

    def __len__(self):
        return len(self.subset)

    def get_class_indices(self):
        class_indices = {}
        for i, sample in tqdm(enumerate(self.subset)):
            label = sample["y"].item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
        return class_indices
    def get_collate_fn(self):
        return None
class FewShotEvalFunc(object):


    def __init__(self,support_data,query_data,metric,pred_name,label_name,k_shot,num_sampling=10):
        self.metric=metric
        self.pred_name=pred_name
        self.label_name=label_name
        self.k_shot=k_shot
        self.num_sampling=num_sampling
        self.support_data=support_data
        self.query_data=query_data


    # def pre_cal_feature(self,evaler:EvalTool):
    #     query_dataloader= evaler.get_dataloader(self.support_data)
    #     device = evaler.device
    #     model= evaler.model
    #     all_new_samples=[]
    #     with torch.no_grad():
    #         for batch in tqdm(query_dataloader): 
    #             batch_gpu=Mapper(lambda x: x.to(device))(batch)
    #             # print(batch[self.label_name])
    #             mvrec_batch=model.get_mvrec(batch_gpu)

    #             # 组合成新的样本
    #             new_samples = [{'y': y, 'mvrec': mvrec} for  y, mvrec in zip( batch["y"], mvrec_batch)]
    #             all_new_samples.extend(new_samples)
    #             # 在这里您可以进一步处理新的样本，例如保存到新的数据集中
    #     self.support_data=MyDataset(all_new_samples)

    def pre_cal_feature(self, evaler:EvalTool):
        # Generate a filename based on dataset characteristics
        def load_or_cal_data( dataset, filepath, evaler):
            # Check if the file exists
            if os.path.exists(filepath):
                print(f"Loading existing samples from {filepath}")
                return MyDataset( torch.load(filepath))
                
            else:
                query_dataloader = evaler.get_dataloader(dataset)
                device = evaler.device
                model = evaler.model
                all_new_samples = []
                with torch.no_grad():
                    for batch in tqdm(query_dataloader):
                        batch_gpu = Mapper(lambda x: x.to(device))(batch)
                        mvrec_batch = model.get_mvrec(batch_gpu)

                        new_samples = [{'y': y, 'mvrec': mvrec} for y, mvrec in zip(batch["y"], mvrec_batch)]

                        all_new_samples.extend(new_samples)
                
                # Save the new_samples to a dataset file
                if not os.path.exists(os.path.dirname(filepath)): os.makedirs(os.path.dirname(filepath))
                torch.save(all_new_samples, filepath)
                print(f"Saved new samples to {filepath}")
                return MyDataset(all_new_samples)
  
        from lyus.Frame import Experiment
        dataset_name= Experiment().get_param().data.data_name
        mv_method= Experiment().get_param().data.mv_method
        clip_name= Experiment().get_param().ClipModel.clip_name
        backbone_name= Experiment().get_param().ClipModel.backbone_name
        path_to_save="./buffer"
        q_or_s="support"
        support_filename = f"{mv_method}/{clip_name}_{backbone_name}_{dataset_name}_{q_or_s}.pt"
        self.support_data =load_or_cal_data(self.support_data,os.path.join(path_to_save, support_filename),evaler=evaler)

        q_or_s="query"
        query_filename = f"{mv_method}/{clip_name}_{backbone_name}_{dataset_name}_{q_or_s}.pt"
        self.query_data =load_or_cal_data(self.query_data,os.path.join(path_to_save, query_filename),evaler=evaler)


    def __call__(self, evaler:EvalTool):
    
        self.pre_cal_feature(evaler=evaler)

        model= evaler.model
        # if evaler.dataloader:
        #     res,cm_img=self.eval_once(model,evaler.dataloader,evaler)
        #     print(res)
        #     cm_img.save(os.path.join( evaler.save_dir,"cm_clip.jpg"))
       
        results,cm_imgs=[],[]

        from lyus.Frame import Experiment
 
        support_dataset_list,query_dataset_list=gen_few_shot_support_query_dataset(self.support_data,self.k_shot,self.num_sampling )
        for  index in  range(len(support_dataset_list)):
            support_dataset=support_dataset_list[index]  #index
     
            Experiment().set_attr("sampling_id",index)
            if  self.query_data: # 如果没有在evaler设置全局的query data，就用采用的
                query_dataloader= evaler.get_dataloader(self.query_data)
            else:
                query_dataset=query_dataset_list[index] #index
                query_dataloader= evaler.get_dataloader(query_dataset)

            if support_dataset:
                model.set_img_prototype(self.k_shot,support_dataset)
            # mulit_view=Experiment().get_param().debug.mulit_view

            res={}
            mv_res,mv_cm_img=self.eval_once(model,query_dataloader,evaler,mulit_view=True)
            res["mv"]=mv_res
            womv_res,womv_cm_img=self.eval_once(model,query_dataloader,evaler,mulit_view=False)
            res["womv"]=womv_res
            mv_cm_img.save(os.path.join( evaler.save_dir,f"cm_{index}.jpg"))
            res= merge_dicts_with_prefixes(res)
            results.append(res)
            if self.k_shot==0:
                break
        # Calculate mean and std of all metrics
        mean_std_dict = calculate_mean_and_std(results)

        print(results)
        print(mean_std_dict)
        self.visual_model_event=True
        return mean_std_dict


    def eval_once(self,model,query_dataloader, evaler:EvalTool,mulit_view):
        res_variables={}
        model.set_mode("infer") # lyus
        from lyus.Frame import Experiment

        def eval_run():
            device = evaler.device
            self.metric.reset()
            with torch.no_grad():
                for batch in tqdm(query_dataloader):
                    # evaler.show_img_tensor(batch)
                    batch_gpu=Mapper(lambda x: x.to(device))(batch)
                    # print(batch[self.label_name])
                    fx=model(batch_gpu,mulit_view)[self.pred_name]

                    # if not hasattr(self,"visual_model_event"):
                    #     evaler.visual_model(batch_gpu)
                    

                    # embed_batch, logits_batch = model(x_batch)
                    # score_batch = 1 - torch.sigmoid(logits_batch)[:, 0]
                    # print(fx,batch["y"])
                    # print(fx, batch[self.label_name])

                    self.metric.add_batch(fx.cpu().numpy(),batch[self.label_name].numpy().astype(np.int64))

            return self.metric.get(),self.metric.visualConfusionMatrix()
    
        metrics, cm_img= eval_run()

        for key ,val in metrics.items():
            res_variables[key]=val
        return res_variables,cm_img











# from lyus.data.folder_dataset import  FolderDataset
# def eval_process(exper: FM.Experiment):
#     exper.info("running ")
#     # return
#     PARAM = exper.get_param()
#     SAVE_DIR = exper.get_save_dir()


#     test_sample_process = ClassficationSampleProcss(PARAM.data.img_size,
#                                                     PARAM.data.means, PARAM.data.stds,withlabel=False)
#     test_data = FolderDataset("/media/lyushuai/Data/DATASET/multi_garde_2",[".png",".jpg"],test_sample_process)
#     test_data=DatasetWrapper(test_data,{"x":"img", "filepath": "filepath","subfolder":"subfolder"})

#     model = eval(PARAM.model.option)(**PARAM.model[PARAM.model.option])

#     ck=CheckPoint(os.path.join(exper.get_save_dir(), "checkpoint"), "OrdNetwork")
#     ck.bind_model(model)
#     ck.load_the_last()

#     model.set_mode("eval")
#     means=PARAM.data.means
#     stds=PARAM.data.stds
#     evalTool=EvalTool(PARAM.device,os.path.join(SAVE_DIR,"visual"),means,stds).load(model)


#     evalTool.run_eval(test_data,EvalFunc())