from random import  shuffle

from torch.utils.data import Sampler

class BalanceSampler(Sampler):
    def __init__(self, data_source,label_loc,num_samples=None,shuffle=True,):
        self.shuffle=shuffle
        self.data_source=data_source
        self._num_samples=num_samples
        self.labels=[   sample[label_loc] for sample in data_source]
        self.classes=set(self.labels)
        self.cls_idx_map={cls:[] for cls in self.classes}
        for idx, cls in enumerate( self.labels):
            self.cls_idx_map[cls].append(idx)


    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __len__(self):
        return self.num_samples


    def __iter__(self):
        cls_idx_map=self.cls_idx_map.copy()
        if self.shuffle:
            for cls,idxs in self.cls_idx_map.items(): shuffle(idxs)
        indices = []
        step=0
        while (True):
            for cls, idxs, in cls_idx_map.items():
                iidx = step % len(idxs)
                if iidx == 0 and self.shuffle:
                    shuffle(cls_idx_map[cls])
                idx = cls_idx_map[cls][iidx]
                indices.append(idx)
                if len(indices)>=self.num_samples:
                    return iter(indices)
            step+=1


# if __name__=="__main__":
#
#     data=[[1,1,3],[1,1,2],[1,1,3],[1,1,3],[1,1,2],[1,1,2],[1,1,2]]
#     sample=BalanceSample(data,2,15)
#     for i in iter(sample):
#         print(data[i][2])
#


