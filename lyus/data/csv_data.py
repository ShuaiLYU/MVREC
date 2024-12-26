import pandas as pd


class CsvLabelData(object):
    
    
    def __init__(self,csv_path,sample_process=None):
        self.csv_path=csv_path
        self.samples=self.csv_to_sample_list(csv_path)
        self.sample_process=sample_process
        
    def csv_to_df(self):
        return pd.read_csv(self.csv_path)
    
    def csv_to_sample_list(self,csv_path):
        data=pd.read_csv(csv_path)
        return data.apply(pd.Series.to_dict, axis=1).to_list()
    
    def get(self):
        return self.samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        if self.sample_process is None:
            return self.samples[idx]
        else:
            return self.sample_process(self.samples[idx])
        
    
    def get_wightSampler_wight(self,key):
        from collections import Counter
        all_value= [ sam[key] for sam in self.samples ]
        # print(self.cls_dict)
        value_num=dict( Counter(all_value))
        num_total=len(self.samples)
        weights=[]
        # print(len(self))
        for data in self.samples:
            w=num_total/value_num[data[key]]
            weights.append(w)
        return weights
