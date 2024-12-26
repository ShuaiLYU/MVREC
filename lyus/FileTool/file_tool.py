
import os
import shutil
import os
from functools import partial

from .utils import list_folder

class Folder(object):
    
    
    def __init__(self, root):
        self.root=root

    def get_child_folders(self):
        return next(os.walk(self.root))[1]


    def exists(self,filename):
        return os.path.exists(os.path.join(self.root,filename))

    def find_files_by_suffix(self,suffixes,recursion=False):
        
        def condition_func(filename,suffix):
            return filename.endswith(suffix)
        
        if not  isinstance(suffixes,(list,tuple)):
            suffixes=[suffixes]
        res=[]
        for suffix in suffixes:
            condition=partial(condition_func,suffix=suffix)
            res+=list_folder(self.root,True,condition,recursion)
        return res

    def find_child_folders(self,condition=None):
        
        dirs=[ {"root":root,"dirs":dirs,"files":files }  for root, dirs, files in os.walk(self.root)][0]["dirs"]
        if condition is not None:
            dirs=[ d for d in dirs if condition(d)]
        dirs= [ os.path.join(self.root,d) for d in dirs]
        return dirs

    def find_file(self,filename):
        for root, _, _ in os.walk(self.root):
            filepath=os.path.join(root,filename)
            if os.path.exists(filepath):
                return filepath
        return None

    def copy_to(self,new_dir):
        shutil.copytree(self.root,new_dir)


    """
    
    os.remove(path)   #删除文件
    os.removedirs(path)   #删除空文件夹
    
    os.rmdir(path)    #删除空文件夹
    
    shutil.rmtree(path)    #递归删除文件夹，即：删除非空文件夹
    """
    def _deep_delete_file(self, func):
        for root, dirs, files in os.walk(self.root):
            for name in files:
                if(func(name)):
                    os.remove(os.path.join(root, name))
                    print("delect : {}  in  {} ".format(name,root))

    def  _deep_delete_folder(self, folder_name):
        for root, dirs, files in os.walk(self.root):
            for dir in dirs:
                if(dir==folder_name):
                    cur_path=os.path.join(root,dir)
                    shutil.rmtree(cur_path)
                    print("remove : {}".format(cur_path))

class FileTool(object):
    
    def __init__(self,filepath):
        
        self.filepath=filepath
        
        if not os.path.exists(self.filepath):
            print("[{}] not exist".format(self.filepath))
        # os.path.ex
        
    def copy_to_dir(self,dst_dir):
        
        dst_name=os.path.basename(self.filepath)
        self.copy_to(os.path.join(dst_dir,dst_name))
        
        
    def copy_to(self,dst_path):
        src=self.filepath
        dst_dir=os.path.dirname(dst_path)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        try:
            shutil.copyfile(src,dst_path)
        except IOError as e:
            print("IOError, when copy {}".format(src))
            return
        except:
            print("unkown error, when copy {}".format(src))
            return 
    
    
    
    def move_to_dir(self,dst_dir):
        
        dst_name=os.path.basename(self.filepath)
        self.move_to(os.path.join(dst_dir,dst_name))
        
        
    def move_to(self,dst_path):
        src=self.filepath
        dst_dir=os.path.dirname(dst_path)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        try:
            shutil.move(src,dst_path)
        except IOError as e:
            print("IOError, when copy {}".format(src))
            return
        except:
            print("unkown error, when copy {}".format(src))
            return 
    
    