import cv2 as cv
import numpy as np
import os
import random


class DefectiveGenerator(object):
    def __init__(self,dir_Database,shape_Img,Limit_ROI,withDatabase=True):
        """

        :param dir_Database:  缺陷ROI路径
        :param shape_Img:   图片大小[height,width]
        :param Limit_ROI:   ROI外接矩形大小[lower ,upper]
        :param withDatabase:   true:从硬盘读入ROI  false：算法生成ROI
        """
        self.dir_Database=dir_Database
        self.height_Img = shape_Img[0]
        self.width_Img=shape_Img[1]
        self.lowerLimit_ROI=Limit_ROI[0]
        self.upperLimit_ROI=Limit_ROI[1]
        #从数据库读入ROI
        self.names_ROIs,self.num_ROIs=self.loadROIs(self.dir_Database)
        if self.num_ROIs<1:
            print("the dataset is empty!")
    def loadROIs(self,dir):
        ROIs=os.listdir(dir)
        num_ROI=len(ROIs)
        return ROIs,num_ROI

    def genDefect(self,img):
        ROI=self.randReadROI()
        ROI_new=self.randMoveROI(ROI)
        #Rows,Cols = np.nonzero(ROI_new)
        #随机设置灰度值
        rand=random.randint(0,200)
        img_rand=self.genRandImg(rand,20,[self.height_Img, self.width_Img])
        img_new=img.copy()
        img_new=img*(1-ROI_new)+img_rand*ROI_new
        return img_new,ROI_new
    def randReadROI(self):
        while(True):
            rand=random.randint(0,self.num_ROIs-1)
            name_Img=self.names_ROIs[rand]
            img_Label=cv.imread(self.dir_Database+"/"+name_Img,0)
            _,ROI=cv.threshold(img_Label,100,255,cv.THRESH_BINARY)
            if(np.sum(ROI)>5):
                return ROI
    def randMoveROI(self,ROI):
        #求图像的域的大小
        Height_Domain =  self.height_Img
        Width_Domain= self.width_Img
        #求ROI区域的坐标
        Rows,Cols = np.nonzero(ROI)
        #求ROI区域的外接矩形大小
        Width_ROI=np.max(Cols)-np.min(Cols)
        Height_ROI=np.max(Rows)-np.min(Rows)
        #随机设置ROI的起始坐标
        Row_Upleft=random.randint(0,Height_Domain-Height_ROI-1)
        Col_Upleft = random.randint(0, Width_Domain - Width_ROI-1)
        Rows=Rows-np.min(Rows)+Row_Upleft
        Cols=Cols-np.min(Cols)+Col_Upleft
        ROI_new=np.zeros([Height_Domain,Width_Domain])
        ROI_new[Rows,Cols]=1
        return ROI_new

    def genRandImg(self,mean,fluct,size):

        low=mean-fluct+(mean-fluct<0)*abs(mean-fluct)
        height=mean+fluct-(mean+fluct>255)*abs(255-(mean+fluct))
        img=np.random.randint(low,height,size)
        img=img.astype("uint8")
        return img

# if __name__=="__main__":
#     fig = plt.figure()
#
#     img=cv.imread("Part4.jpg",0)
#     shape_Img=img.shape
#     print(shape_Img)
#     gen=DefectiveGenerator("./label",shape_Img,[0,10000])
#     new_img=gen.genDefect(img)
#     plt.subplot(121),plt.imshow(img,'gray'),plt.title('img')
#     plt.subplot(122), plt.imshow(new_img, 'gray'), plt.title('new img')
#     plt.show()


