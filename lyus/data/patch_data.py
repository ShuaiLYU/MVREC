
from PIL  import  Image
import  random,os
import  numpy as np
"""

patch  base dataset 

"""

class PatchImg(object):

    def __init__(self,img, h_patch, w_patch):
        self.img=img
        self.img_w, self.img_h=self.img.width,self.img.height
        self.h_patch,self.w_patch=h_patch, w_patch
        self.all_patch_array=self.get_all_patch_idx(self.img_w,self.img_h,self.w_patch,self.h_patch)
        # print(self.all_patch_array.shape)


    def get_patches(self,step_w,step_h):
        patch_array = self.group_patches_by_step(self.all_patch_array, step_w, step_h)
        return patch_array

    def get_patched_imgs(self,step_w,step_h):
        patch_array = self.group_patches_by_step(self.all_patch_array, step_w, step_h)
        shape=(patch_array.shape[0],patch_array.shape[1],self.h_patch,self.w_patch)
        img_patch_array=np.empty(shape=shape)
        for j in range(patch_array.shape[0]):
            for i in range(patch_array.shape[1]):
                img_patch_array[j,i]=self.get_patch_img(self.img,patch_array[j,i])
        return img_patch_array

    def load_img(self,img_path):
        img=Image.open(img_path)
        return img


    @staticmethod
    def get_patch_img(img,patch):
        x,y,w,h=patch
        left, upper, right, lower=x,y,x+w,y+h
        return img.crop(box=(left, upper, right, lower))


    """
        get all possible patch loc;
    """
    @staticmethod
    def get_all_patch_idx(w_img,h_img, w_patch,  h_patch):
        Xs, Ys = w_img - w_patch+1, h_img - h_patch+1
        patch_array=np.zeros(shape=(Ys,Xs,4))

        for i,x in enumerate(range(Xs)):
            for j,y in enumerate(range(Ys)):
                patch_array[j][i]=np.array([x,y,w_patch,h_patch])
        return patch_array


    """
        devide all patches into step_w*step_h groups
    """
    @staticmethod
    def group_patches_by_step(patch_array,step_w,step_h,x_group_idx=0,y_group_idx=0):
        if x_group_idx==None:  x_group_idx=np.random.randint(0,step_w)
        if y_group_idx == None: y_group_idx = np.random.randint(0, step_h)
        assert x_group_idx<step_w and y_group_idx<step_h
        Xs=patch_array.shape[1]//step_w+(patch_array.shape[1]%step_w>0)
        Ys=  patch_array.shape[0]//step_h+(patch_array.shape[0]%step_h>0)

        patch_array_group = np.zeros(shape=(Ys, Xs, 4))
        for i,x in enumerate(range(Xs)):
            for j, y in enumerate(range(Ys)):
                loc_x = x_group_idx + x * step_w
                loc_y = y_group_idx + y* step_h
                loc_x,loc_y=min(patch_array.shape[1]-1,loc_x),min(patch_array.shape[0]-1,loc_y)
                patch_array_group[j][i] = patch_array[loc_y,loc_x]
        return   patch_array_group




    def get_patches_per_img(self,idx,num_patches_per_img,h_img, w_img, h_patch, w_patch, h_step=1, w_step=1
                          ):
        """
        :param idx:  输入图像的唯一标识
        :param h_img:  输入图像的高度
        :param w_img:  输入图像的宽度
        :param h_patch:  图像块的高度
        :param w_patch:  图像块的宽度
        :param h_step:    高度方向步长
        :param w_step:      宽度方向步长
        :param num_patch_each_img:  随机采样多少个Patch(不放回)，如果Patch不足则不采样
        :return:  Patches: list(patch)   patch=(x,y,w_patch,h_patch)
        """

        X_limit, Y_limit = w_img - w_patch, h_img - h_patch
        Xs, Ys = range(0, X_limit + 1, w_step), range(0, Y_limit + 1, h_step)
        # print(Xs)
        patches = list()
        for x in Xs:
            for y in Ys:
                patch = (x, y, w_patch, h_patch,idx)
                assert x + w_patch <= w_img, y + h_patch <= h_img
                patches.append(patch)
        # print(len(patches))



        if (len(patches) > num_patches_per_img):
            patches = random.sample(patches, num_patches_per_img)
        return patches


if __name__ == '__main__':

    img=np.zeros((501,501))
    b=PatchImg(img,500,500)
    patch_arr=b.get_patched_imgs(500,500)
    print(patch_arr.shape)
    patch_arr=b.get_patches(500,500)



