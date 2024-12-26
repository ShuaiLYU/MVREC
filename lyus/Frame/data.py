


from torch.utils.data import Sampler
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
import  math
from random import  shuffle

try:
    from collections import Iterable

except ImportError:
    from collections.abc import Iterable


class unifiedBatchSampler(object):
    def __init__(self, data_labels,batch_size):

        super(unifiedBatchSampler,self).__init__()
        self.batch_size=batch_size
        self.cls_dict=self.get_cls_dict(data_labels)
        maxNumCls = max([len(val) for _, val in self.cls_dict.items()])
        self.numBatchCls = math.ceil(maxNumCls / float(self.batch_size))
        self.numCls = len(self.cls_dict)
        self.num = self.numCls * self.numBatchCls * self.batch_size

    def get_cls_dict(self,label_list):
        cls_dict = {}
        for idx,label in enumerate(label_list):
            if label not in cls_dict.keys(): cls_dict[label]=[]
            cls_dict[label].append(idx)
        return cls_dict


    def __len__(self):

        return self.num

    def __iter__(self):

        for b in range(self.numBatchCls):
            for cls in range(self.numCls):
                batch = []
                for i in range(self.batch_size):
                    loc=b*self.batch_size+i
                    loc=loc%len(self.cls_dict[cls])
                    if loc==0:
                        shuffle(    self.cls_dict[cls])
                        # print(self.cls_dict[2])
                    yield self.cls_dict[cls][loc]
                    # batch.append(self.cls_dict[cls][loc])

                # yield batch








from torch.utils.data import Dataset
class DatasetWrapper(Dataset):
    
    def __init__(self,dataset, mapping=None,new_length=None,transform=None,sample_process=None):
        
        self.dataset=dataset
        self.mapping=mapping
        self.new_length=new_length
        self.transform=transform
        self.sample_process=sample_process
        if self.new_length is not None:
            self._iidxes=[ i%len(self.dataset) for i  in range(self.new_length)]
        else:
            self._iidxes=[ i%len(self.dataset) for i  in range(len(self.dataset))]
            
    def __len__(self):
        return len(self._iidxes)
    
    def __getitem__(self, idx):
        # print(idx, len(self._iidxes))
        idx=self._iidxes[idx]
        item=self.dataset[idx]
        if self.transform is not None:
            item["img"]=self.transform(item["img"])
        
        # print(item.keys())
        if self.mapping==None:
            pass
        elif  isinstance(self.mapping,tuple) or isinstance(self.mapping,list):
            item= [item[k] for k in self.mapping]
        elif  isinstance(self.mapping,dict):  
            item= { k: item[v] for k, v in self.mapping.items() } 
        else:
            item=item[self.mapping]
        if self.sample_process is not None:
            item=self.sample_process(item)
        return item 
    
    def get_sample_wights(self,key):
        from collections import Counter
        all_value= [ sam[key] for sam in self ]
        # print(self.cls_dict)
        value_num=dict( Counter(all_value))
        num_total=len(self)
        weights=[]
        # print(len(self))
        for data in self:
            w=num_total/value_num[data[key]]
            weights.append(w)
        return weights
    
    def __getattr__(self, name):
        """
        如果尝试访问的属性在这个类中不存在，则尝试从self.dataset中调用。
        """
        return getattr(self.dataset, name)

# def sample_list_to_csv(samples,csvname=None):
#     assert isinstance(samples,list) and len(samples)>0
#     for sam in samples:
#         assert  isinstance(sam,dict)
#     print(samples[0])
#     keys=list(samples[0].keys())
#     data=[]
#     for col in range(len(samples)):
#         sample=samples[col]
#         item=[  sample[keys[i]] for i in range(len(keys) )]
#         data.append(item)

#     data=pd.DataFrame(data,columns=keys)
#     if csvname is not None:
#         data.to_csv(csvname,index=False)
#     return data

        
        
        
class ComposeJoint(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, x):
		for transform in self.transforms:
			x = self._iterate_transforms(transform, x)

		return x
    
    
    
	def append(self, transform):
		self.transforms.append(transform) 
        
	def _iterate_transforms(self, transforms, x):
		if isinstance(transforms, Iterable):
			for i, transform in enumerate(transforms):
				x[i] = self._iterate_transforms(transform, x[i])
		else:

			if transforms is not None:
				x = transforms(x)

		return x
    
    