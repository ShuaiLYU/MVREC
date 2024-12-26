import os
import sys

def list_folder(root,use_absPath=True, func=None,recursion=True):
    """
    :param root:  文件夹根目录
    :param func:  定义一个函数，过滤文件
    :param use_absPath:  是否返回绝对路径， false ：返回相对于root的路径
    :return:
    """
    root = os.path.abspath(root)
    if os.path.exists(root):
        print("遍历文件夹【{}】......".format(root))
    else:
        raise Exception("{} is not existing!".format(root))
    files = []
    # 遍历根目录,
    for cul_dir, _, fnames in sorted(os.walk(root)):
        for fname in sorted(fnames):
            path = os.path.join(cul_dir, fname)#.replace('\\', '/')
            if  func is not None and not func(path):
                continue
            if use_absPath:
                files.append(path)
            else:
                files.append(os.path.relpath(path,root))
        if not recursion:
            break
    print("    find {} file under {}".format(len(files), root))
    return files


class ImageLabelPair(object):

    def __init__(self, root, img_suffix=".png", label_suffix=".json", recursion=False, label_root=None):
        self.root = root
        if label_root is None: label_root = root

        if_label_img = lambda x: x.endswith(label_suffix) and (not x.endswith(img_suffix))
        if_img = lambda x: x.endswith(img_suffix) and (not x.endswith(label_suffix))

        imgs = list_folder(root, False, if_img, recursion=recursion)

        samples = []
        img_with_empty_label = 0
        for img in imgs:
            label = None
            label_name =img.replace(img_suffix, label_suffix)
            label_path = os.path.join(label_root, label_name)
            if os.path.exists(label_path):
                label = label_path
            if label is None: img_with_empty_label += 1
            sam = ( os.path.join(root, img), label)
            samples.append(sam)

        self.samples = samples

        print("{} samples without label file".format(img_with_empty_label))
        # for img ,label in self.samples:
        #     assert(label in labels),label

"""
get the folder name in which  the file is located
获得输入文件名，所在文件夹的名字

"""
def get_folder_name_of_file(file):
    dir_name=os.path.split(os.path.realpath(file))[0]
    return os.path.basename(dir_name)


