import cv2
import numpy as np

def  dye_display(img, heatmap,thresh=100,dilate_size=5,border_color=(0,100,255)):
    # 将图像转换为灰度图像
    # 对图像进行平滑滤波
    heatmap = cv2.GaussianBlur(heatmap, (3, 3), 0)
    heatmap = cv2.normalize(heatmap.astype('float'), None, 0.0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(heatmap, thresh, 255, cv2.THRESH_BINARY)
    
    # # 对二值化掩模进行膨胀
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    # mask = cv2.dilate(mask, kernel, iterations=5)
    # # print(np.unique(mask))
    # # 将掩模转换为彩色图像
    heatmapshow = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # heatmapshow = cv2.GaussianBlur(heatmapshow, (5, 5), 0)
    heatmapshow = cv2.addWeighted(img, 0.8, heatmapshow, 0.2, 0)     # # 将掩模和原始图像融合
    # heatmapshow=heatmapshow*mask[:,:,np.newaxis]+img*(1-mask[:,:,np.newaxis])
    mask3d=np.stack([mask]*3,axis=-1 )
    heatmapshow[mask3d==0]=img[mask3d==0]

    # 寻找前景的边界
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 将边界绘制在原始图像上
    border = cv2.drawContours(img.copy(), contours, -1, border_color, 2)
    # border=img.copy()
    # #染色
    # dye=np.zeros(border.shape,dtype=np.uint8)
    # fill_color=(0,255,100)
    # cv2.fillPoly(dye, pts=contours, color=fill_color)
    


    # # 将原图、掩模、边界和融合结果拼接起来


    return mask,heatmapshow,border



def  gen_heatmap(img, heatmap,thresh=100,dilate_size=5,border_color=(0,100,255)):
    # 将图像转换为灰度图像
    # 对图像进行平滑滤波
    heatmap = cv2.GaussianBlur(heatmap, (3, 3), 0)
    heatmap = cv2.normalize(heatmap.astype('float'), None, 0.0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(heatmap, thresh, 255, cv2.THRESH_BINARY)

    # # 将掩模转换为彩色图像
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # heatmapshow = cv2.GaussianBlur(heatmapshow, (5, 5), 0)
    heatmapshow = cv2.addWeighted(img, 0.8, heatmap, 0.2, 0)     # # 将掩模和原始图像融合
    # heatmapshow=heatmapshow*mask[:,:,np.newaxis]+img*(1-mask[:,:,np.newaxis])
    mask3d=np.stack([mask]*3,axis=-1 )
    heatmapshow[mask3d==0]=img[mask3d==0]


    # # 将原图、掩模、边界和融合结果拼接起来
    return heatmap,heatmapshow