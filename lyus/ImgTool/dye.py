import cv2
import numpy as np
import matplotlib.pyplot as plt


def  dye_display(img, heatmap,thresh=100,dilate_size=5,border_color=(0,100,255)):
    # 将图像转换为灰度图像
    # 对图像进行平滑滤波
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)

    # 对图像进行自动阈值分割，得到二值化掩模
    # _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, mask = cv2.threshold(heatmap, thresh, 255, cv2.THRESH_BINARY)
    # 对二值化掩模进行膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    mask = cv2.dilate(mask, kernel, iterations=5)

    # 将掩模转换为彩色图像
    color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 寻找前景的边界
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fill_color=(0,255,100)
    # 将边界绘制在原始图像上
    border = cv2.drawContours(img.copy(), contours, -1, border_color, 2)
    dye=np.zeros(border.shape,dtype=np.uint8)
    cv2.fillPoly(dye, pts=contours, color=fill_color)
    # # 将掩模和原始图像融合
    result = cv2.addWeighted(border, 0.9, dye, 0.1, 0)

    # # 将原图、掩模、边界和融合结果拼接起来
    # img_concat = np.concatenate((img, color_mask, border, result), axis=1)
    return result



import cv2
import numpy as np
import matplotlib.pyplot as plt
from LyusLib.Frame.utils import InverseNormalize,ToNumpy
from LyusLib.Frame.saver import Visualer
class VisualTool(object):
    
    def __init__(self,mean=None,std=None):
        
        self.inverse=InverseNormalize(mean,std)
        self.toNp=ToNumpy(transpose=True)
        self._num_finished=0
        
    def set_save_dir(self,save_dir):
        self.visualer=Visualer(save_dir)
        
    def visualize(self, batch_list, child_dir, file_name_batch,**kwargs):
        self.visualer.visualize(batch_list, child_dir, file_name_batch,**kwargs)

    def dye(self,x_batch,fx_batches):
  
        num=x_batch.shape[0]
        x_batch=self.inverse(x_batch)
        x_batch=self.toNp(x_batch)
        new_fx_batch=[]
        paintings=[]
        for i in range(num):
            img=x_batch[0]
            fx=fx_batches[0].squeeze()
            # print(img.shape,fx.shape)
            img=img.astype(np.uint8)
            
            #to [0,255]
            fx_img=self.heatmap_to_image(fx)
            fx_img=fx_img.astype(np.uint8)
            new_fx_batch.append(fx_img.squeeze()[:,:,np.newaxis])
            # paint=self.dye_display(img,fx)
            paint=self.gen_heatmap(img,fx)

            # print(x_batch.shape,paint.shape,111)
            paintings.append(paint)
        return x_batch,paintings
    def reset(self):
        self._num_finished=0
        
    def do_it(self,x_batch,y_batch,fx_batches,file_name_batch=None):
        if file_name_batch is None:
            file_name_batch=[ str(self._num_finished +i)+".jpg" for i in range(len(x_batch)) ]
        self._num_finished+=len(x_batch)
        batches=list(self.dye(x_batch,fx_batches))
        
        if y_batch is not None:
            y_batch=np.array(y_batch)
            y_batch=y_batch.squeeze()[np.newaxis,:,:,np.newaxis]
            y_batch=y_batch.astype(np.uint8)*255
            batches=list(batches)+[y_batch]
            # for fx in fx_batches: print(fx.shape)
            # for x in x_batch: print(x.shape)

        # for b in batches: print(np.array(b).shape)
        self.visualize(batches,"",file_name_batch)

    @staticmethod
    def  dye_display(img, heatmap,thresh=200,dilate_size=5,border_color=(255,100,0),weight=0.2):
        # 将图像转换为灰度图像
        # 对图像进行平滑滤波
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)

        # 对图像进行自动阈值分割，得到二值化掩模
        # _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, mask = cv2.threshold(heatmap, thresh, 255, cv2.THRESH_BINARY)
        # 对二值化掩模进行膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        mask = cv2.dilate(mask, kernel, iterations=5)

        # 将掩模转换为彩色图像
        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 寻找前景的边界
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # fill_color=(0,255,100)
        fill_color=border_color
        # 将边界绘制在原始图像上
        border = cv2.drawContours(img.copy(), contours, -1, border_color, 2)
        dye=np.zeros(border.shape,dtype=np.uint8)
        cv2.fillPoly(dye, pts=contours, color=fill_color)
        # # 将掩模和原始图像融合
        result = cv2.addWeighted(border, 1-weight, dye, weight, 0)

        # # 将原图、掩模、边界和融合结果拼接起来
        # img_concat = np.concatenate((img, color_mask, border, result), axis=1)
        return result
    @staticmethod                         
    def heatmap_to_image(heatmap):
        # 将heatmap缩放到[0, 255]范围内
        heatmap = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 将heatmap转换为RGB颜色映射
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap

    @staticmethod 
    def  gen_heatmap(img, heatmap,thresh=150,dilate_size=5,border_color=(0,100,255)):
        # 将图像转换为灰度图像
        # 对图像进行平滑滤波
        heatmap = cv2.GaussianBlur(heatmap, (3, 3), 0)
        heatmap = cv2.normalize(heatmap.astype('float'), None, 0.0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(heatmap, thresh, 255, cv2.THRESH_BINARY)

        # # 将掩模转换为彩色图像
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[...,::-1]
        # heatmapshow = cv2.GaussianBlur(heatmapshow, (5, 5), 0)
        heatmapshow = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)     # # 将掩模和原始图像融合
        # heatmapshow=heatmapshow*mask[:,:,np.newaxis]+img*(1-mask[:,:,np.newaxis])
        mask3d=np.stack([mask]*3,axis=-1 )
        heatmapshow[mask3d==0]=img[mask3d==0]


        # # 将原图、掩模、边界和融合结果拼接起来
        return heatmapshow
            
            
            

if __name__=="__main__":

    root = "C:/Users/shuai/Desktop/temp/heatmap/"
    # 读取原始图片
    img = cv2.imread(root+'img.png')
    mask = cv2.imread(root+'heatmap.png')
    img=cv2.resize(img,(512,512))
    mask=cv2.resize(mask,(512,512))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    res=dye_display(img,mask)
    # 显示结果
    plt.figure(figsize=(15,10))
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
