

import pandas as pd
import json
import csv
from collections import OrderedDict
from functools import partial
import cv2
import os 

import numpy as np
import torch

from   torchvision import transforms


"""
run:
    m10=Mapper(lambda  x: x*10) 
    inputs=[1,2,[1,2],{"a":1,"b":2}]
    print(m10(inputs))
output :
[10, 20, [10, 20], {'a': 10, 'b': 20}]
"""
class Mapper(object):
    def __init__(self, func):
        # initialize the object with a function
        self.func = func

    def __call__(self, objects):
        # apply the function to different types of objects

        # if the objects are a tuple, then apply the function to each element and return a tuple
        if isinstance(objects, tuple):
            return tuple(self(obj) for obj in objects)
        # if the objects are a list, then apply the function to each element and return a list
        elif isinstance(objects, list):
            return [self(obj) for obj in objects]
        # if the objects are a dictionary, then apply the function to each value and return a dictionary
        elif isinstance(objects, dict):
            return {k: self(v) for k, v in objects.items()}
        # otherwise, apply the function to the objects directly and return the result
        else:
            return self.func(objects)

class DictExecutor(object):
    def __init__(self,func):
        # initialize the object with a function
        self.func=func
    def __call__(self,kv:dict):
        # iterate over the key-value pairs in the dictionary
        for key, val  in kv.items():
            # if the value is another dictionary, then flatten it and call recursively
            if isinstance(val, dict):
                subdict={  "_".join([key,subk]):subv for subk ,subv in val.items()}
                self(subdict)
            # otherwise, apply the function to the key-value pair
            else:
                self.func(key,val)

        
class InverseNormalize(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
    
    
to_cpu=Mapper(lambda x: x.detach().cpu()) 
    
class ToNumpy(object):
    
    """
    map a tensor ranging [0,1]  to a uin8 numpy array ranging [ 0,255]
    
    """
    def __init__(self,multiplier=255,dtype=np.uint8,transpose=False,squeeze=False):
        
        self.multiplier=multiplier
        self.dtype=dtype
        self.transpose=transpose
        self.squeeze=squeeze
    def run(self, tensor:torch.tensor):
        
        if self.transpose:
            assert(len(tensor.shape)==4),"tensor dims must be 4  when setting transpose as True !!! "
        tensor=tensor*self.multiplier
        if tensor.is_cuda: tensor=tensor.cpu()
        array=tensor.numpy().astype(self.dtype)
        if self.transpose: array=array.transpose(0,2,3,1)
        if self.squeeze: array=array.squeeze()
        return array
    def __call__(self, data):
        return Mapper(self.run)(data)
    
import os
class Folder(object):
    
    
    def __init__(self, root):
        
        self.root=root

    def exists(self,filename):
        
        return os.path.exists(os.path.join(self.folder,filename))
    
    
    def find_file(self,filename,recursion=False):
        pass
    
    def find_files_by_suffix(self,suffixes,recursion=False):
        
        def condition_func(filename,suffix):
            return filename.endswith(suffix)
        
        if not  isinstance(suffixes,(list,tuple)):
            suffixes=[suffixes]
        res=[]
        for suffix in suffixes:
            condition=partial(condition_func,suffix=suffix)
            res+=Folder.list_folder(self.root,True,condition,recursion)  
        return res
    
    
    def find_child_folders(self,condition=None):
        
        dirs=[ {"root":root,"dirs":dirs,"files":files }  for root, dirs, files in os.walk(self.root)][0]["dirs"]
        if condition is not None:
            dirs=[ d for d in dirs if condition(d)]
        dirs= [ os.path.join(self.root,d) for d in dirs]
        return dirs
        
    
    
    @staticmethod
    def list_folder(root, use_absPath=True, func=None, recursive=True):
        """
        :param root:  文件夹根目录
        :param func:  定义一个函数，过滤文件
        :param use_absPath:  是否返回绝对路径， false ：返回相对于root的路径
        :return:
        """
        root = os.path.abspath(root)
        if os.path.exists(root):
            print("遍历文件夹【{}】......".format(root))
        else:
            raise Exception("{} is not existing!".format(root))
        files = []
        # 遍历根目录,
        for cul_dir, _, fnames in sorted(os.walk(root)):

            for fname in sorted(fnames):
                path = os.path.join(cul_dir, fname)  # .replace('\\', '/')
                if func is not None and not func(path):
                    continue
                if use_absPath:
                    files.append(path)
                else:
                    files.append(os.path.relpath(path, root))
            if not recursive: break
        print("    find {} file under {}".format(len(files), root))
        return files

    
    

    
    
    
class CsvLogger(object):
    
    
    def __init__(self,root,csv_name):
        suffix=".csv"
        self.root=root
        self.csv_name=csv_name if csv_name.endswith(suffix) else csv_name+suffix
        self.csv_path=os.path.join(self.root,self.csv_name)
        
        if not  os.path.exists(self.root):
            os.makedirs(self.root)
        self.header=None
        self.rows=[]
        if os.path.exists(os.path.join(self.root,self.csv_name)):
            self.header,self.rows=self._read_csv(self.csv_path)
            # print(111,self.csv_path,self.header,rows)
            # print("found a existing csv file and load the header: {}...".format(self.header))
        # if not Folder(self.root).exists(csv_name):
        
    def get_cols(self,key):
        return  [  row[key] for row in self.rows]
    
    def set_header(self,header:list):
        
        assert(self.header==None)
        self.header=header    

        self._append_one_row({ k:k for k in self.header})
        return self
    
    def get_rows(self):
        _,rows=self._read_csv(self.csv_path)
        return rows
    
    # def _read_csv(self,csv_path):
    #     with open(csv_path, newline='') as csvfile:
    #         spamreader = [  row for row  in csv.reader(csvfile, delimiter=',', quotechar='|') ]
    #         header=spamreader[0]
    #         rows=[]
    #         for row_val in spamreader[1:]:
    #             rows.append({ key:val for key,val in zip(header,row_val)})
    #         return header,rows
    def _read_csv(self,csv_path):
        import pandas as pd
        df_data = pd.read_csv(csv_path)
        # print(df_data,csv_path)
        header= df_data.columns.to_list()
        rows=[ row.to_dict() for  idx, row in df_data.iterrows()]
        return header,rows
    
    def _save_csv(self, rows):
        with open(self.csv_path, 'w', newline='') as csvfile:
            csvw = csv.writer(csvfile)
            csvw.writerow(self.header)
            for row in rows:
                filled_row = {k: row.get(k, '') for k in self.header}
                csvw.writerow([filled_row[k] for k in self.header])
        
    def append_one_row(self,row:dict,strict=True,check_header=True):
        if check_header and self.header is None:
            self.set_header(list(row.keys()))
        self._append_one_row(row,strict)

    def _append_one_row(self,row:dict,strict=True):

        if strict:
            assert(len(row)==len(self.header))
            missing_keys = [k for k in row.keys() if k not in self.header]
            assert not missing_keys, f"Missing keys in header: {missing_keys}"
            row= [  row[k] for k in  self.header]
     
            with open(self.csv_path, 'a+', newline='') as csvfile:
                print(row)
                csvw = csv.writer(csvfile)
                csvw.writerow(row)
        else:
            missing_keys = [k for k in row.keys() if k not in self.header]
            if missing_keys:
                self.header.extend(missing_keys)
            self.rows.append(row)
            self._save_csv(self.rows)

    
    def get_average_results(self,key_name="data_name",val_name="acc",condition:dict=None,add_keys=[]):
        import pandas as pd
        import ast
        from collections import defaultdict

        # 读取CSV文件get

        df = pd.read_csv(self.csv_path)
        # 初始化一个字典，用于保存所有键的总和
        sum_dict = defaultdict(lambda: defaultdict(float))
        count_dict = defaultdict(int)
        sum_numeric = defaultdict(float)

        def is_number(num):
            try:
                float(num)
                return True
            except (ValueError , TypeError,SyntaxError):
                return False

        # 对每一行的每一个单元格中的字典或数字进行求和
        old_rows=[]
        for index, row in df.iterrows():
            cond= [ row[kk]==vv  for kk,vv in condition.items() ]
            # print(cond)
            if  not all(cond): continue

            old_rows.append( row.to_dict())
            for col in df.columns:
                if pd.notna(row[col]):  # 检查单元格是否为空
                    try:
                        # 尝试将字符串转换为字典
                        cell_dict = ast.literal_eval(row[col])
                        if isinstance(cell_dict, dict):
                            for k, v in cell_dict.items():
                                if not is_number(v): continue
                                sum_dict[col][k] += v
                            count_dict[col] += 1
                        else:
                            raise ValueError("Unexpected value type")
                    except (ValueError, SyntaxError):
                        # 如果单元格不是字典，则假定为数字
                        try:
                            sum_numeric[col] += float(row[col])
                            count_dict[col] += 1
                        except ValueError:
                            pass
                            # raise ValueError(f"Unexpected value type in column {col}, row {index}")
        # print(sum_numeric,count_dict)
        # 计算平均值并生成新的字典
        avg_dict = {}
        for col in df.columns:
            if col in sum_dict and sum_dict[col]:
                avg_dict[col] = {k: v / count_dict[col] for k, v in sum_dict[col].items()}
            elif col in sum_numeric and count_dict[col]:
                avg_dict[col] = sum_numeric[col] / count_dict[col]

        # 将平均值字典或数值添加到DataFrame的最后一行
        new_row = {}
        for col in df.columns:
            if col in avg_dict:
                if isinstance(avg_dict[col], dict):
                    new_row[col] = str(avg_dict[col])
                else:
                    new_row[col] = avg_dict[col]
            else:
                new_row[col] = None  # 如果列没有数据，填充为空值

        new_row["data_name"]="average"
        # print(new_row)


        res_row={}
        for kk,vv in condition.items(): res_row[kk]=vv
        for add_key in add_keys:  res_row[add_key]=old_rows[0][add_key]
        for row in old_rows+[new_row]:
            key= row[key_name]
            val = ast.literal_eval(row[val_name])["mean"]
            # val= 
            res_row[key]=round(val,4)
        return res_row
    

import time
def get_current_time_point():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())




def get_class_name(x):
    return x.__class__.__name__



"""
get the folder name in which  the file is located
获得输入文件名，所在文件夹的名字

"""
def get_folder_name_of_file(file):
    dir_name=os.path.split(os.path.realpath(file))[0]
    return os.path.basename(dir_name)

