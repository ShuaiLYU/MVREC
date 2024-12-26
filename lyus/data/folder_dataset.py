


from .utils import list_folder
import os

def get_subfolder(root):
    subfolders = [os.path.join(root, folder) for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder))]
    return subfolders

class FolderDataset(object):


    def __init__(self, root,suffixes:list=None,sample_process=None,level=1):

        assert level==1 or level ==2
        self.sample_process=sample_process
        self.samples=[]
        def check_func(filename):
            if not suffixes:
                return True
            for suffix in suffixes:
                if filename.endswith(suffix):
                    return True
            return  False
        if level==1:
            subfolders = get_subfolder(root)
            for subfolder in subfolders:
                filenames=list_folder( subfolder,True,check_func)
                for filename in filenames:
                    self.samples.append({"subfolder": os.path.basename(subfolder), "filepath": filename})
        else:
            for subfolder in get_subfolder(root):
                subfolder_name= os.path.basename(subfolder)
                for subsubfolder in get_subfolder(subfolder):
                    subsubfolder_name= os.path.basename(subsubfolder)
                    filenames=list_folder( subsubfolder,True,check_func)
                    for filename in filenames:
                        self.samples.append({"subfolder": subfolder_name,"subsubfolder": subsubfolder_name, "filepath": filename})
                            

    def __len__(self):
        return len(self.samples)


    def __getitem__(self,idx):
        if self.sample_process is None:
            return self.samples[idx]
        else:
            return self.sample_process(self.samples[idx])

