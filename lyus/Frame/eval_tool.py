from tqdm    import  tqdm
from torch.utils.data import DataLoader
import  torch
from lyus.Frame.utils import  InverseNormalize,ToNumpy,Mapper
from lyus.Frame.saver import Visualer
import  numpy as np
from PIL import  Image,ImageDraw,ImageFont
import os 

import datetime

def generate_filename_with_timestamp(extension=".jpg"):
    # 获取当前日期和时间
    now = datetime.datetime.now()
    # 将当前日期时间格式化为字符串
    # 例如: '2023-10-05_12-30-59'
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    # 拼接文件名和文件扩展名
    filename = f"{timestamp}{extension}"
    return filename
class EvalTool(object):

    def __init__(self,device,save_dir,mean,std,**kwargs):
        self.device=device
        def get_dataloader(dataset):
            return DataLoader(dataset, batch_size=kwargs.get("batch_size", 8),
                        shuffle=False,
                        num_workers=kwargs.get("num_workers", 8),collate_fn=dataset.get_collate_fn(),
                        drop_last=False)
        self.get_dataloader=get_dataloader
        self.inverse = InverseNormalize(mean, std)
        self.toNp = ToNumpy(transpose=True)
        self.save_dir=save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.visualer=Visualer(save_dir)
        def to_device(data):
            if isinstance(data,str):
                return data
            return data.to(self.device)


        self.to_device=   Mapper(to_device)
        self.visualer=Visualer(save_dir)
        def to_cpu(data):
            if isinstance(data,str):
                return data
            return data.detach().cpu()
        self.to_cpu=Mapper(to_cpu)



    def show_img_tensor(self, inputs,img_key="x",label_key="y"):
        inputs=self.to_cpu(inputs)
        img_tensor=inputs[img_key]
        cls_array=inputs[label_key].numpy()
        bboxes_array=inputs["bboxes"].numpy()
        # tempsetbegin
        assert len(img_tensor.shape)==5
        num_view=img_tensor.shape[1]
        img_tensor=img_tensor.reshape(-1, *img_tensor.shape[2:] )
        bboxes_array=bboxes_array.reshape(-1, *bboxes_array.shape[2:] )
        cls_array=np.repeat(cls_array[:,np.newaxis],num_view,1).flatten()

        # tempsetend


        img_array=self.toNp(self.inverse(img_tensor))
        num = img_array.shape[0]
        imgs = []
        pictures = []
        filenames=[]
        childfolders=[]
        visualer=self.visualer
        for i in range(num):
            img = img_array[i].astype(np.uint8)
            cls = int(cls_array[i])
            bbox=bboxes_array[i][0]
     
            # filename=os.path.basename(filename_batch[i])
            filename=generate_filename_with_timestamp()
            subfolder=str(cls)
            # text="score:{:.3f},grade:{:.3f}".format(score,grade)
            pic=self.show_text_on_img(256,img,"text",bbox=bbox.tolist())
            imgs.append(img)
            pictures.append(pic)
            filenames.append(filename)
            childfolders.append(subfolder)
        visualer.visualize([imgs],childfolders,filenames)
        childfolders =[ "show/"+subdir for subdir in childfolders ]
        visualer.visualize([ pictures], childfolders, filenames)



    def load(self,model):
        if hasattr(self,"model"):
            del self.model
            torch.cuda.empty_cache()
        self.model=model
        self.model.to(self.device)
        return self


    def run_eval(self,iter_func):

        return iter_func(self)

    @staticmethod
    def show_text_on_img(pic_sz, img, text,bbox=None, font_sz=20):
        def get_linux_font_path():
            import os, subprocess
            assert (os.name == "posix")
            font_name = subprocess.check_output(['fc-match', '-f', '%{file}']).decode('utf-8').strip()
            return font_name

        if not isinstance(img, Image.Image):
            img=Image.fromarray(img)
        front_rows = pic_sz // 5
        img_h = pic_sz - front_rows
        img_w = pic_sz

        picture = Image.new('RGB', (pic_sz, pic_sz), (20, 20, 20))
        draw = ImageDraw.Draw(picture)
        draw.rectangle([0, 0, pic_sz, front_rows], fill=0)
 

        # 在图像的前 20 行像素上显示文本
        # font = ImageFont.truetype('arial.ttf', 15)
        # font=ImageFont.load_default().font_variant(size=font_sz)
        # font = ImageFont.truetype("Arial.ttf", size=15)

        font = ImageFont.truetype(get_linux_font_path(), font_sz)
        text_width, text_height = draw.textsize(text, font)
        draw.text(((pic_sz - text_width) / 2, (front_rows - text_height) / 2), text, fill='white', font=font)

        original_image = img

        if bbox:
            draw = ImageDraw.Draw(original_image)
            draw.rectangle(bbox,width=2) 


        original_width, original_height = original_image.size
        original_ratio = original_width / original_height
        target_ratio = img_w / img_h

        if original_ratio > target_ratio:
            # 原始图像过宽，需要按照宽度缩放
            new_width = img_w
            new_height = int(new_width / original_ratio)
        else:
            # 原始图像过高，需要按照高度缩放
            new_height = img_h
            new_width = int(new_height * original_ratio)

        original_image = original_image.resize((new_width, new_height))

        # 将原始图像粘贴到新图像的中心位置
        x = (pic_sz - new_width) // 2
        y = (pic_sz + front_rows - new_height) // 2
        picture.paste(original_image, (x, y))
        return picture

    @staticmethod
    def float2str(x:float,palce=3):
        return format(x, f'.{palce}f')
    

    def visual_model(self,inputs):
        from torch.utils.tensorboard import SummaryWriter
        import torchvision.models as models

        # 创建一个 TensorBoard summary writer
        writer = SummaryWriter( self.save_dir+'/model_visualization')

        # 获取一个模型
        model =self.model

        # 使用 writer 添加模型
        writer.add_graph(model, inputs,use_strict_trace=False)
        writer.close()
