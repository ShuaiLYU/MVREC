

from torch.utils.tensorboard import SummaryWriter
#  tensorboard --logdir=runs

# from lyu_libs.plt_utils import colors_map
import matplotlib.pyplot as plt
import  numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os , random

class TBLoader(object):
    def __init__(self,log_dir):
        assert os.path.exists(log_dir),"{} is not existing!!!".format(log_dir)
        self.log_dir=log_dir
        self.ea = event_accumulator.EventAccumulator(log_dir)
        self.ea.Reload()
        self.Keys= self.ea.scalars.Keys()

    def get_vals(self,key):
        assert key in self.Keys,"get an unexcepted key:{}".format(key)
        vals = [i.value for i in self.ea.scalars.Items(key)]
        return vals

    def get_dics(self,key):
        assert key in self.Keys,"get an unexcepted key:{}".format(key)
        dics = { i.step:i.value for i in self.ea.scalars.Items(key) }
        return dics

    def get_walltimes(self,key):
        assert key in self.Keys,"get an unexcepted key:{}".format(key)
        times = [i.wall_time for i in self.ea.scalars.Items(key)]
        return times

    def get_steps(self,key):
        assert key in self.Keys,"get an unexcepted key:{}".format(key)
        steps = [i.step for i in self.ea.scalars.Items(key)]
        return steps

    def save_to_csv(self,log_dir=None):
        if log_dir is None:
            log_dir=self.log_dir

        #列索引
        cols=self.get_steps(self.Keys[0])
        #行索引   #数据
        rows,datas=[],[]

        ##添加walltimes
        rows.append("walltimes")
        datas.append(self.get_walltimes(self.Keys[0]))

        ##添加数据
        for key in self.Keys:
            data=self.get_vals(key)
            rows.append(key)
            datas.append(data)
        #写入
        self.to_csv(datas,csv_path=log_dir+"/res.csv",cols=cols,rows=rows,transpose=True)


    @staticmethod
    def to_csv(mat, csv_path, rows=[], cols=[],transpose=False):
        mat = np.array(mat)
        assert mat.ndim == 2,"nidm:{} should be 2!".format(mat.ndim)
        if transpose:
            mat=mat.T
            rows,cols=cols,rows
        r, c = mat.shape
        assert r == len(rows) and c == len(cols)
        dic = {cols[i]: mat[:, i] for i in range(c)}
        columns = list(dic.keys())
        data = pd.DataFrame(dic,
                            columns=cols,
                            index=rows)
        data.to_csv(csv_path, header=True, index=True)

    # @staticmethod
    # def imshow(data,savefig=False, title="tittle",xlabel="xlabel",ylable="ylabel",**kwargs):
    #     plt.rc('font', family='Times New Roman')
    #     # plt.figure()
    #     # set(gca, 'linewidth', 2, 'fontsize', 30, 'fontname', 'Times')
    #     r, c = np.array(data).shape
    #     xlim=kwargs.get("xlim",None)
    #     if xlim is not None:
    #         plt.xlim(*xlim)  # 限定横轴的范围
    #     ylim = kwargs.get("ylim", None)
    #     if xlim is not None:
    #         plt.ylim(*ylim)  # 限定纵轴的范围
    #     colors=kwargs.get("colors",list(colors_map.keys()))
    #     x = range(1, c+1)
    #     cnt = range(1, r + 1)
    #     for i in cnt:
    #         y = data[i - 1]
    #         plt.plot(x, y, mec='r', color=colors[i-1], mfc='w', )  # marker='o'
    #     # font = {'family':'Times New Roman','weight': 'normal','size': 23}
    #     # plt.legend()  # 让图例生效
    #     xticks = [0] + [i for i in x if i % 5 == 0]
    #     plt.xticks(xticks, xticks, )
    #     plt.margins(0)
    #     plt.subplots_adjust(bottom=0.15)
    #     plt.xlabel(xlabel)  # X轴标签
    #     plt.ylabel(ylable)  # Y轴标签
    #     plt.title(title)  # 标题
    #     if not savefig:
    #         plt.show()
    #     else:
    #         print(title)
    #         print("save to:{}".format(title+".jpg"))
    #         plt.savefig(title+".jpg")
    #     plt.close()
