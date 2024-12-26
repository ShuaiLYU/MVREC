##########################################
from pathlib import Path
from PIL import Image, ImageDraw
import random
import numpy as np


def rect2bbox(rect):
    x, y, w, h = rect
    return (x, y, x + w, y + h)


class PatchImg(object):

    def __init__(self, img_h, img_w, h_patch, w_patch):
        self.img_w, self.img_h = img_w, img_h
        self.h_patch, self.w_patch = h_patch, w_patch
        self.all_patch_array = self.get_all_patch_idx(self.img_w, self.img_h, self.w_patch, self.h_patch)
        # print(self.all_patch_array.shape)

    def get_patches(self, step_w, step_h, x_group_idx=0, y_group_idx=0):
        patch_array = self.group_patches_by_step(self.all_patch_array, step_w, step_h, x_group_idx, y_group_idx)
        return patch_array

    def get_patched_imgs(self, img, step_w, step_h):
        def get_image_channel(img: Image):
            assert (img.mode == "L" or img.mode == "RGB")
            return len(img.getbands())

        patch_array = self.group_patches_by_step(self.all_patch_array, step_w, step_h)
        img_ch = get_image_channel(img)

        shape = (patch_array.shape[0], patch_array.shape[1], self.h_patch, self.w_patch, img_ch)
        img_patch_array = np.empty(shape=shape)
        for j in range(patch_array.shape[0]):
            for i in range(patch_array.shape[1]):
                img_patch_array[j, i] = self.get_patch_img(img, patch_array[j, i])
        return patch_array, img_patch_array

    def load_img(self, img_path):
        img = Image.open(img_path)
        return img

    def load_img(self, img_path):
        img = Image.open(img_path)
        return img

    @staticmethod
    def show_patches_on_image(img, rect_array):
        imgDraw = ImageDraw.Draw(img)
        for j in range(rect_array.shape[0]):
            for i in range(rect_array.shape[1]):
                bbox = rect2bbox(rect_array[j][i])
                imgDraw.rectangle(bbox, outline='purple', width=2)
        return img

    @staticmethod
    def get_patch_img(img, patch):
        x, y, w, h = patch
        left, upper, right, lower = x, y, x + w, y + h
        return img.crop(box=(left, upper, right, lower))

    """
        get all possible patch loc;
    """

    @staticmethod
    def get_all_patch_idx(w_img, h_img, w_patch, h_patch):
        Xs, Ys = w_img - w_patch + 1, h_img - h_patch + 1
        patch_array = np.zeros(shape=(Ys, Xs, 4))

        for i, x in enumerate(range(Xs)):
            for j, y in enumerate(range(Ys)):
                patch_array[j][i] = np.array([x, y, w_patch, h_patch])
        return patch_array

    """
        devide all patches into step_w*step_h groups
        x_group_idx=0,y_group_idx=0


    """

    @staticmethod
    def group_patches_by_step(patch_array, step_w, step_h, x_group_idx=0, y_group_idx=0):
        if x_group_idx == None:  x_group_idx = np.random.randint(0, step_w)
        if y_group_idx == None: y_group_idx = np.random.randint(0, step_h)
        assert x_group_idx < step_w and y_group_idx < step_h
        Xs = patch_array.shape[1] // step_w + (patch_array.shape[1] % step_w > x_group_idx)
        Ys = patch_array.shape[0] // step_h + (patch_array.shape[0] % step_h > y_group_idx)

        patch_array_group = np.zeros(shape=(Ys, Xs, 4))
        for i, x in enumerate(range(Xs)):
            for j, y in enumerate(range(Ys)):
                loc_x = x_group_idx + x * step_w
                loc_y = y_group_idx + y * step_h
                loc_x, loc_y = min(patch_array.shape[1] - 1, loc_x), min(patch_array.shape[0] - 1, loc_y)
                patch_array_group[j][i] = patch_array[loc_y, loc_x]
        return patch_array_group

    def get_patches_per_img(self, idx, num_patches_per_img, h_img, w_img, h_patch, w_patch, h_step=1, w_step=1
                            ):
        """
        :param idx:      输入图像的唯一标识
        :param h_img:    输入图像的高度
        :param w_img:    输入图像的宽度
        :param h_patch:  图像块的高度
        :param w_patch:  图像块的宽度
        :param h_step:   高度方向步长
        :param w_step:      宽度方向步长
        :param num_patch_each_img:  随机采样多少个Patch(不放回)，如果Patch不足则不采样
        :return:  Patches: list(patch)   patch=(x,y,w_patch,h_patch,idx)
        """

        X_limit, Y_limit = w_img - w_patch, h_img - h_patch
        Xs, Ys = range(0, X_limit + 1, w_step), range(0, Y_limit + 1, h_step)
        # print(Xs)
        patches = list()
        for x in Xs:
            for y in Ys:
                patch = (x, y, w_patch, h_patch, idx)
                assert x + w_patch <= w_img, y + h_patch <= h_img
                patches.append(patch)
        # print(len(patches))

        if (len(patches) > num_patches_per_img):
            patches = random.sample(patches, num_patches_per_img)
        return patches


"""

    patch_array,img_patch_array=PatchImg(img.size[1],img.size[0],256,256).get_patched_imgs(img,256,256)
    img_name=os.path.basename(sam[0])
    for col in range(patch_array.shape[0]):
        for row in range(patch_array.shape[1]):
            rect=patch_array[col][row].tolist()
            bbox=rect2bbox(rect)
            img_patch=Image.fromarray(img_patch_array[col][row].astype(np.uint8))
            patch_name=img_name.split(".")[0]+"_"+bbox2str(bbox)+"."+img_name.split(".")[1]
            img_patch.save(os.path.join(save_dir,patch_name))

            if with_mask:
                mask_patch=mask.crop(box=bbox)
                patch_mask_name=patch_name.split(".")[0]+"_mask."+patch_name.split(".")[1]
                mask_patch=mask_patch.convert("P")
                # palette=np.array([[0,0,0],[0,255,0],[0,255,0]]).astype(np.uint8)

                palette=np.array([[255,0,0],[0,255,0],[0,0,255], [255,255,0]])
                palette=np.concatenate([np.zeros((1,3)),palette],axis=0).astype(np.uint8)
                mask_patch.putpalette(palette)
                mask_patch.save(os.path.join(save_dir,patch_mask_name))

"""
#############################################################


