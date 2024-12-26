import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from random import  shuffle
import  math
import numpy as np
class DataManager(object):
    def __init__(self, dataset,param,shuffle=True):
        """
        """
        self.shuffle=shuffle
        self.dataset=dataset
        self.data_size=len(dataset)
        self.epochs_num=10
        self.batch_size = param["batch_size"]
        self.next_batch=self.get_next()
        self._session= tf.Session()
        self.num_batch = param.get("num_batch_train", -1)
        self.num_batch=math.ceil(self.data_size / self.batch_size) if self.num_batch== -1 else self.num_batch
    def get_next(self):
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.int32,tf.int32, tf.string))
        dataset = dataset.repeat(self.epochs_num)
        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size*3)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch

    def generator(self):
        while True:
            for index in range(self.data_size):
                yield self.dataset[index]

    def __iter__(self):
        self.cnt_batch=0
        return self
    def __next__(self):
        if self.cnt_batch < len(self):
            self.cnt_batch+=1
            next_batch = self._session.run(self.next_batch)
            return next_batch
        else:
            raise  StopIteration

    def __len__(self):
        return  self.num_batch


class DataManager_balance(DataManager):
    def __init__(self, dataset, param):
        shuffle = param.get("shuffle", True)
        super(DataManager_balance,self).__init__(dataset, param, shuffle)
        self.num_batch = param.get("num_batch_train", 0)
        if self.num_batch ==-1:
            self.num_batch =len(self.dataset.cls_dict[1])*len(self.dataset.cls_dict)//param["batch_size"]
        if self.num_batch ==0:
            self.num_batch =len(self.dataset.cls_dict[0])*len(self.dataset.cls_dict)//param["batch_size"]
        self.unified_batch=param.get("unified_batch",False)
        if self.unified_batch:
            print("using unified_batch...")
            self.next_batch_pos=self.get_next_pos()
            self.next_batch_neg=self.get_next_neg()
    def get_next(self):
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.int32,tf.int32, tf.string))
        dataset = dataset.repeat(self.epochs_num)
        # if self.shuffle:
        #     dataset = dataset.shuffle(self.batch_size*3+200)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch

    def generator(self):
        step=0
        cls_idxs_map={ key:list(range(len(val))) for key,val in self.dataset.cls_dict.items()}
        while(True):
            for cls, idxs, in cls_idxs_map.items():
                iidx=step%len(idxs)
                if iidx==0 and self.shuffle:
                    shuffle(cls_idxs_map[cls])
                idx = cls_idxs_map[cls][iidx]
                yield self.dataset.getitem_cls(idx,cls)
            step+=1

    def get_next_pos(self):
        dataset = tf.data.Dataset.from_generator(self.generator_pos, (tf.float32, tf.int32,tf.int32, tf.string))
        dataset = dataset.repeat(10)
        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size*3)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch

    def generator_pos(self):
        while True:
            for index in range(len(self.dataset.cls_dict[1])):
                yield self.dataset.getitem_cls(index,1)


    def get_next_neg(self):
        dataset = tf.data.Dataset.from_generator(self.generator_neg, (tf.float32, tf.int32,tf.int32, tf.string))
        dataset = dataset.repeat(10)
        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size*3)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch

    def generator_neg(self):
        while True:
            for index in range(len(self.dataset.cls_dict[0])):
                yield self.dataset.getitem_cls(index, 0)

    def __next__(self):
        if not self.unified_batch:
            return super().__next__()
        if self.cnt_batch < len(self):
            if self.cnt_batch%2==0:
                next_batch = self._session.run(self.next_batch_pos)
            else:
                next_batch = self._session.run(self.next_batch_neg)
            self.cnt_batch += 1
            return next_batch
        else:
            raise  StopIteration



class DataManager_valid(object):
    def __init__(self,dataset,batch_size):
        """
        """
        self.dataset=dataset
        self.data_size=len(dataset)
        self.batch_size =batch_size
        self.num_batch=math.ceil(self.data_size/self.batch_size)

    def get_a_batch(self,cnt_batch):
        assert  cnt_batch<self.num_batch
        idx_begin=cnt_batch*self.batch_size
        idx_end=min((cnt_batch+1)*self.batch_size,self.data_size)
        batch=[ self.dataset[idx] for idx in range(idx_begin,idx_end)]
        num_items=len(self.dataset[0])
        def get_items(batch,idx):
            return np.array([ sample[idx] for sample in batch])
        return  [ get_items(batch,idx)  for idx in range(num_items)]
    def __len__(self):
        return self.num_batch
    def __iter__(self):
        self.cnt_batch=0
        return self
    def __next__(self):
        if   self.cnt_batch<self.num_batch:
            cul_batch=self.get_a_batch(self.cnt_batch,)
            self.cnt_batch+=1
            return cul_batch
        else:
          raise StopIteration  # 一定有终止