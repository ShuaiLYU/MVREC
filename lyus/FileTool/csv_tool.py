

import numpy as np 

import os
import csv

import pandas as pd



# 保留 小数

def  keep_decimal_places(num,places):
    if isinstance(num,float):
        return round(num,places)
    else:
        return num



# 转置


def transpose(data):
    return data.T




from typing import List
from typing import Union



def df_from_dicts(data:List[dict]):
    return pd.DataFrame(data)

def df_to_dicts(data):
    return  [ row for idx, row in data.iterrows()]

class CsvReader(object):
    def __init__(self,csv_path):
        self.csv_path=csv_path
        self.load()

    def load(self):
        def _read_csv(csv_path):
            import pandas as df
            df_data = df.read_csv(csv_path)
            # df_data = df.read_csv(csv_path,index_col=0)
            
            header= df_data.columns.to_list()
            rows=[ row.to_dict() for  idx, row in df_data.iterrows()]
            # print(header,rows)
            return header,rows
        if self.exists():
            self.header,self.rows=_read_csv(self.csv_path)
            
        else:
            self.header,self.rows=[],[]
    
    
    def get_last_row(self):
        return self.rows[-1]
        
    def get_rows(self):
        rows=self.rows
        return rows
    
    def exists(self):
        return os.path.exists(self.csv_path)
    
class CsvLogger(CsvReader):
    
    
    def __init__(self,root,csv_name):
        suffix=".csv"
        self.root=root
        self.csv_name=csv_name if csv_name.endswith(suffix) else csv_name+suffix
        self.csv_path=os.path.join(self.root,self.csv_name)
        
        if not  os.path.exists(self.root):
            os.makedirs(self.root)
        super(CsvLogger,self).__init__(self.csv_path)
        
        if self.exists():
            self.flag=0
        else:
            self.flag=1
    
    def if_new(self):
        return self.flag
    
    
    def set_header(self,header:list):
        assert(self.header==None)
        self.header=header    
        self.append_one_row({ k:k for k in self.header})
        return self
    
    
    def update_header(self,row:dict):
        self.load()
        # print(self.header,self.rows)
        if len(row)==len(self.header) and all([ (k in self.header) for k,v in row.items()]):
            return 
        for k,v in row.items():
            if k not in self.header:
                self.header.append(k)


        with open(self.csv_path, 'w+', newline='') as csvfile:
            csvw = csv.writer(csvfile)
            csvw.writerow(self.header)
            if len(self.rows)>0:
                for new_row in self.rows:
                    for k in self.header:
                        if k not in new_row.keys():
                            new_row[k]=None
                    csvw.writerow([ new_row[k] for k in self.header])

            
    
    def append_one_row(self,row:dict,strict=False):
        self.update_header(row)
        row= [  row[k] if k in row.keys() else None for k in  self.header]
        with open(self.csv_path, 'a+', newline='') as csvfile:
            csvw = csv.writer(csvfile)
            csvw.writerow(row)

################pandas

def get_max_row_by(df,key):
    ind = df[key].idxmax()
    # row = df.iloc[ind,:]
    return ind




class CsvTool(object):
    @staticmethod
    def get_all_csv(root, keys=None):
        assert (isinstance(keys,list) or isinstance(keys,str))
        if isinstance(keys,str):
            keys=[keys]
        csv_names = [name for name in os.listdir(root) if name.endswith(".csv")]
        return_names=[]
        for csv_name in csv_names:
            ifwith= [ key in csv_name for key in keys ]
            if(all(ifwith)):
                return_names.append(csv_name)

        return [os.path.join(root, csv) for csv in return_names]
    @staticmethod
    def get_all_rows_from_csvs(csv_paths):
        all_data = []
        for csv_path in csv_paths:
            exp_name = os.path.basename(csv_path)
            csv = CsvReader(csv_path)
            datas = csv.rows
            for data in datas: data["exp_name"] = exp_name
            all_data += datas
        return all_data


