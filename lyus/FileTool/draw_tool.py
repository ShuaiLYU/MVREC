# import pandas as pd

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

class SurSeries(object):
    
    def __init__(self,param_name,value_name,params:list,values:list,conditions:dict):
        
        assert(len(params)==len(values))
        self.param_name=param_name
        self.params=params
        self.values=values
        self.conditions=conditions
        self.value_name=value_name
        
    def get_items(self):
        
        items=[]
        
        for i  in range(len(self.params)):
            item=self.conditions.copy()
            item[self.param_name]=self.params[i]
            item[self.value_name]=self.values[i]
            items.append(item)
        return items
    
    
    
    
    @staticmethod
    def draw_SurSeries(data,x,y,hue, colors=["g","b","r"],**kwargs):


        hue_names=[]

        total_data=[]
        for d in data:
            items=d.get_items()
            # print(items)
            total_data += items
            hue_names.append(d.conditions[hue])

        total_data=pd.DataFrame(total_data)

        # plt.rcParams.update({'legend.fontsize':12})
        # print(hue_names)
        palette ={   hue_name:colors[idx] for idx,hue_name in enumerate(hue_names) }
        print(palette)
        linestyles=kwargs.get("linestyles",["-", "--","dashed"])
        markers=kwargs.get("markers",["^", "o","."])
        capsize=kwargs.get("capsize",0.1)
        figsize=kwargs.get("figsize",None)
        linewidth=kwargs.get("linewidth",None)
        draw_grid=kwargs.get("draw_grid",True)
        fontsize=kwargs.get("fontsize",15)
        ylim=kwargs.get("ylim",None)
        ax=kwargs.get("ax",None)
        def draw(data):
            if figsize is not None: plt.figure(figsize=figsize)
       
            
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            if ylim is not None:
                plt.ylim(ylim)
   
            # # 设置坐标标签字体大小
            if ax is not None:
                ax.set_xlabel(..., fontsize=fontsize,fontweight='bold')
                ax.set_ylabel(..., fontsize=fontsize,fontweight='bold')
            # 设置图例字体大小
            # ax.legend(..., fontsize=fontsize)

            sns.pointplot(x=x, y=y,hue=hue,data=data,palette=palette,
            markers=markers, linestyles=linestyles,capsize=capsize,ax=ax)
            if draw_grid: plt.grid()
            # plt.legend(...,fontsize=fontsize)
            if  kwargs.get("show",True):plt.show() 
        draw(total_data)


        
    @staticmethod
    def draw_barplot(data,x,y,hue, colors=["g","b","r"],**kwargs):


        hue_names=set()

        total_data=[]
        for d in data:
            items=d.get_items()
            # print(items)
            total_data += items
            hue_names.add(d.conditions[hue])

        total_data=pd.DataFrame(total_data)

        # plt.rcParams.update({'legend.fontsize':12})
        # print(hue_names)
        palette ={   hue_name:colors[idx] for idx,hue_name in enumerate(hue_names) }
        linestyles=kwargs.get("linestyles",["-", "--","dashed"])
        markers=kwargs.get("markers",["^", "o","."])
        capsize=kwargs.get("capsize",0.1)
        figsize=kwargs.get("figsize",None)
        draw_grid=kwargs.get("draw_grid",True)
        fontsize=kwargs.get("fontsize",15)
        ylim=kwargs.get("ylim",None)
        ax=kwargs.get("ax",None)
        def draw(data):
            if figsize is not None: plt.figure(figsize=figsize)

            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            if ylim is not None: plt.ylim(ylim)
            # ax = plt.subplot(111)
            # # 设置坐标标签字体大小
            # 设置图例字体大小
            # ax.legend(..., fontsize=fontsize)

            # sns.pointplot(x=x, y=y,hue=hue,data=data,palette=palette,markers=markers, linestyles=linestyles,capsize=capsize)
            handle=sns.barplot(data=data,x=x, y=y, hue=hue,palette=palette,ax=ax)

            if draw_grid: plt.grid()
            # plt.legend(...,fontsize=fontsize)
            if  kwargs.get("show",True):plt.show() 
            return handle
            
        return draw(total_data)

        
        
        

#