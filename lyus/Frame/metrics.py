import numpy as  np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from PIL import Image
import io

__all__ = ['classificationMetric',"SegmentationMetric","OrdinalClassMetric"]

from .base import MetricBase,HookBase

from .utils import Mapper

from  lyus.Frame import Experiment

class CommonMetricHook(HookBase):

    def __init__(self ,dataset ,metric ,data_name ,pred_name,period ,batch_size=64,label_name="y",):

        self.dataset =dataset
        self.data_name =data_name
        self.pred_name=pred_name
        self.metric =metric
        self.period =period
        self.label_name=label_name
        if not  isinstance(dataset,DataLoader ):
            self.data_loader= DataLoader(self.dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=0 ,drop_last=False)
        else:
            self.data_loader=dataset

    def after_epoch(self):

        epoch = self.trainer.epo
        if epoch %self.period!=0:
            return
        model= self.trainer.model
        model.set_mode("test")
        metrics= self.eval_run(model)
        # for k ,val in metrics.items():
        #     self.trainer.epo_variables[self.metric_name+k]=val
        epo_name="_".join(["epo",str(self.trainer.step)])
        for key ,val in metrics.items():
            key = "_".join([self.data_name, key])
            self.trainer.epo_variables[key]=val
            Experiment().add_scalar(key, val, self.trainer.step)
            Experiment().info("_".join([epo_name,str(key) ,str(val),]))

    def eval_run(self, model):
        device = self.trainer.device
        self.metric.reset()
        with torch.no_grad():
            for batch in self.data_loader:
                batch_gpu=Mapper(lambda x: x.to(device))(batch)
                fx=model(batch_gpu)[self.pred_name]
                # embed_batch, logits_batch = model(x_batch)
                # score_batch = 1 - torch.sigmoid(logits_batch)[:, 0]
                # print(fx,batch["y"])
                self.metric.add_batch(fx.cpu().numpy(), batch[self.label_name].numpy().astype(np.int64))

        return self.metric.get()


# class MetricHook(HookBase):
#
#     def __init__(self,dataset,metric,period,batch_size=64):
#
#         self.dataset=dataset
#         self.metric=metric
#         self.period=period
#         self.data_loader= DataLoader(self.dataset, batch_size=batch_size,
#                                     shuffle=False, num_workers=0,drop_last=False)
#
#
#     def after_epoch(self):
#
#         epoch = self.trainer.epo
#         if epoch%self.period!=0:
#             return
#
#         self.eval_func()
#
#
#         model.train()
#         return self.metric.get()
#
#
#     def eval_func(self):
#         model= self.trainer.model
#         model.eval()
#         self.metric.reset()
#         for x_batch,y_batch in self.data_loader:
#             fx_batch=model(x_batch)
#             self.metric.addBatch(fx_batch,y_batch)
#
        

class AucMetric(MetricBase):
    def add_batch(self, scores, labels):
        labels = np.where(labels >= 1, 1, 0)
        self.scores.append(scores)
        self.labels.append(labels)
        
    def get(self):  
        res={}
        fpr, tpr, _ = roc_curve(np.concatenate(self.labels), np.concatenate(self.scores))
        roc_auc = auc(fpr, tpr)
        # res["fpr"]=fpr
        # res["tpr"]=tpr
        res["roc_auc"]=roc_auc
        return res




from sklearn.metrics import f1_score
import numpy as np
class BinaryClassMetric(MetricBase):

    def add_batch(self, scores, labels):
        labels = np.where(labels >= 1, 1, 0)
        self.scores.append(scores)
        self.labels.append(labels)
    def get(self):  
        res={}
        best_f1, best_threshold, error_count, fp, fn,P,N= self.compute_f1_threshold(
            np.concatenate(self.labels),  np.concatenate(self.scores))
        res["bin_best_f1"]=best_f1
        res["bin_best_threshold"]=best_threshold
        res["bin_error_count"]=error_count
        res["bin_fp"]=fp
        res["bin_fn"]=fn
        res["bin_P"]=P
        res["bin_N"]=N
        return res
        
    @staticmethod
    def compute_f1_threshold(labels, scores):
        """
        计算F1 score、最优阈值、错误分类数、FP和FN
        :param labels: 一维数组，元素为0或1，表示真实标签
        :param scores: 一维数组，元素为实数，表示分类器的预测得分
        :return: F1 score, 最优阈值, 错误分类数, FP, FN
        """
        # 计算F1 score和最优阈值
        f1_scores = []
        thresholds = np.unique(scores) # 尝试所有可能的阈值
        # print(thresholds)
        for t in thresholds:
            f1 = f1_score(labels, scores > t)
            f1_scores.append(f1)
        f1_scores = np.array(f1_scores)
        best_f1 = f1_scores.max()
        best_threshold = thresholds[f1_scores.argmax()]

        # 将预测得分转换为预测标签，并计算错误分类数、FP和FN
        predicted_labels = (scores > best_threshold).astype(int)
        error_count = np.sum(predicted_labels != labels)
        fp = np.sum(np.logical_and(predicted_labels == np.ones_like(labels), labels == np.zeros_like(labels)))
        fn = np.sum(np.logical_and(predicted_labels == np.zeros_like(labels), labels == np.ones_like(labels)))
        P,N=np.sum(labels == np.ones_like(labels)),np.sum(labels == np.zeros_like(labels))
        
        return best_f1, best_threshold, error_count, fp, fn,P,N
# compute_f1_threshold([1,0,1,0,1],[0.5,0.2,0.89,0.8,0.4])


class classificationMetric(MetricBase):

    def __init__(self,numClass,ignore_list=None):

        self.labels=[]
        self.scores = []
        self.ignore_list =ignore_list
        self.numClass=numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
        print("classificationMetric, cls: {}".format(self.numClass))


    def reset(self):
        self.labels=[]
        self.scores=[]
        self.confusionMatrix=np.zeros((self.numClass, self.numClass))
        return self
    def add_batch(self,scores,labels):
        # print(np.unique(labels))
        if not len(scores.shape)==2:
            scores=scores[None]
        assert labels.shape==scores.shape[:-1],f"{labels.shape}, {scores.shape}"
        mask=~np.isin(labels,self.ignore_list)
        # print(scores)
        if mask.size==0: return
        labels=labels[mask]
        scores=scores[mask]
        self.labels.append(labels)
        self.scores.append(scores)
        predicts=np.argmax(scores,axis=-1)
        cm=self.genConfusionMatrix(predicts,labels)
        self.confusionMatrix+=cm
        # print(cm)



    def get(self) -> dict:
        res = {}
        res["acc"] = round(self.Accuracy(), 4)
        res["meanPrecision"] = round(self.meanPrecision(), 4)
        res["meanRecall"] = round(self.meanRecall(), 4)

        return res


    def genConfusionMatrix(self, Predict, Label):
        # print(Predict,Label)
        # remove classes from unlabeled pixels in gt image and predict
        mask = (Label >= 0) & (Label < self.numClass)
        label = self.numClass * Label[mask] + Predict[mask]

        count = np.bincount(label.astype(np.int8), minlength=self.numClass**2)
        cm = count.reshape(self.numClass, self.numClass)
        return cm

    def visualConfusionMatrix(self, figsize=(8, 6)):
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        # Create a heatmap
        sns.heatmap(self.confusionMatrix.astype(np.int16) , annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        # Convert buffer to PIL.Image
        img = Image.open(buf).convert("RGB")
        return img



    def Accuracy(self):
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc

    def classPrecision(self):
        # return each category  accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        return np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)

    def meanPrecision(self):
        classPre = self.classPrecision()
        meanPre = np.nanmean(classPre)
        return meanPre

    def classRecall(self):
        # print(self.confusionMatrix)
        return np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
    
    def meanRecall(self):
        class_Recal = self.classRecall()
        mean_recall = np.nanmean(class_Recal)
        return mean_recall
    
    def save_cm_map(self,save_path,axis_name=None,dpi=200):
        fig = plt.figure()
        df=pd.DataFrame(self.confusionMatrix.astype(np.int),index=axis_name,columns=axis_name)
        fig=sns.heatmap(df,annot=True)
        fig.get_figure().savefig(save_path, dpi = dpi)

class Accucary(classificationMetric):
    def get(self) -> dict:
        res = {}
        res["acc"] = round(self.Accuracy(), 4)
        return res
 



class OrdinalClassMetric(classificationMetric):

    def __init__(self,numClass):

        self.labels=[]
        self.scores = []
        self.numClass=numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
        print("ordinal class, cls: {}".format(self.numClass))
    def reset(self):
        self.labels=[]
        self.scores=[]
        self.confusionMatrix=np.zeros((self.numClass, self.numClass))
        return self
    @staticmethod
    def round_to_nearest_integer(array, min_val, max_val):
        rounded_array = np.around(array)
        clamped_array = np.clip(rounded_array, min_val, max_val)
        return clamped_array

    def add_batch(self,scores,labels):
        self.labels.append(labels)
        self.scores.append(scores)
        predicts=self.round_to_nearest_integer(scores,0,self.numClass-1)
        cm=self.genConfusionMatrix(predicts,labels)
        self.confusionMatrix+=cm
    def get(self) ->dict:
        res={}
        res["ord_acc"]=self.Accuracy()
        res["bin_ord_acc"]=self.bin_accuracy()
        res["ord_mae"] = self.mean_absolute_error()
        print(res)
        return res

    def mean_absolute_error(self):
        labels = np.concatenate(self.labels)
        scores = np.concatenate(self.scores)
        mae = np.mean(np.abs(scores - labels))
        return mae

    def bin_accuracy(self):
        bin_cm=np.array( [ [self.confusionMatrix[0,0].sum(), self.confusionMatrix[0,1:].sum()],
                           [self.confusionMatrix[1:,0].sum(),self.confusionMatrix[1:,1:].sum()]
                           ])
        print(bin_cm)
        acc = np.diag(bin_cm).sum() /  bin_cm.sum()
        return acc
# class OrdinalClassMetric(classificationMetric):
#
#     def __init__(self,numClass):
#
#         self.labels=[]
#         self.scores = []
#         self.numClass=numClass
#         self.confusionMatrix = np.zeros((self.numClass,)*2)
#         print("ordinal class, cls: {}".format(self.numClass))
#
#
#     def reset(self):
#         self.labels=[]
#         self.scores=[]
#         self.confusionMatrix=np.zeros((self.numClass, self.numClass))
#         return self
#     def add_batch(self,scores,labels):
#         assert labels.shape==scores.shape[:-1]
#         self.labels.append(labels)
#         self.scores.append(scores)
#         predicts=np.argmax(scores,axis=-1)
#         cm=self.genConfusionMatrix(predicts,labels)
#         self.confusionMatrix+=cm
#
#
#     def get(self) ->dict:
#         res={}
#         res["acc"]=self.Accuracy()
#         res["mae"] = self.mean_absolute_error()
#
#         return res
#
#     def mean_absolute_error(self):
#         labels = np.concatenate(self.labels)
#         # scores=[ np.argmax(score,axis=-1) for score in self.scores  ]
#         scores = np.concatenate(self.scores)
#         mae = np.mean(np.abs(scores - labels))
#         return mae

class SegmentationMetric(MetricBase):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

        
        
    def addBatch(self, imgPredict, imgLabel):
        #print(imgPredict.shape, imgLabel.shape)
        assert imgPredict.shape == imgLabel.shape,"imgPredict shape:{}" "imgLabel shape:{}".format(imgPredict.shape,imgLabel.shape)

        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
        
        
    def get(self)->dict:
        res={}
        res["pixelAccuracy"]=self.pixelAccuracy()
        res["meanIntersectionOverUnion"]=self.meanIntersectionOverUnion()
        return res
        
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def clsIntersectionOverUnion(self,cls):
        assert cls<self.numClass
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        return IoU[cls]

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU


