
from PIL import Image
import numpy as np
import os

def concatImage(images,mode="Adapt",scale=0.5,offset=None,background=100):
	"""
	:param images:  图片列表
	:param mode:     图片排列方式["Row" ,"Col","Adapt"]
	:param scale:
	:param offset:    图片间距
	:return:
	"""
	if not isinstance(images, list):
		raise Exception('images must be a  list  ')
	if mode not in ["Row" ,"Col","Adapt"]:
		raise Exception('mode must be "Row" ,"Adapt",or "Col"')
	images=[np.uint8(img) for  img in images]   # if Gray  [H,W] else if RGB  [H,W,3]
	images = [img.squeeze(2)  if   len(img.shape)>2 and img.shape[2]==1 else img for img in images]
	count = len(images)
	img_ex = Image.fromarray(images[0])
	size=img_ex.size #[W,H]
	if mode=="Adapt":
		mode= "Row" if size[0]<=size[1] else "Col"
	if offset is None:offset = int(np.floor(size[0] * 0.02))
	if mode=="Row":
		target = Image.new(img_ex.mode, (size[0] * count+offset*(count-1), size[1] * 1),background)
		for i  in  range(count):
			image = Image.fromarray(images[i]).resize(size, Image.BILINEAR).convert(img_ex.mode)
			target.paste(image, (i*(size[0]+offset), 0))
			#target.paste(image, (i * (size[0] + offset), 0, i * (size[0] + offset) + size[0], size[1]))
		return target
	if mode=="Col":
		target = Image.new(img_ex.mode, (size[0] , size[1]* count+offset*(count-1)),100)
		for i  in  range(count):
			image = Image.fromarray(images[i]).resize(size, Image.BILINEAR).convert(img_ex.mode)
			target.paste(image, (0,i*(size[1]+offset)))
			#target.paste(image, (0, i * (size[1] + offset), size[0], i * (size[1] + offset) + size[1]))
		return target

def visualization(list_batchs,filenames,save_dirs):
	"""
	:param list_batchs:  list[  array[b,h,w,c] ,array[b,h,w,c] ]  or  [   [imags],[images]  ]
	:param filenames:		list [filename]
	:param save_dir:
	:return:
	"""
	#batch_num= filenames.shape(0)
	for i, filename in enumerate(filenames):
		if not isinstance(save_dirs, list):
			#save_dirs = [save_dirs for i in len(batch_num)]
			save_dir = save_dirs
		else:
			save_dir=save_dirs[i]
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		if not isinstance(filename,str): filename=filename.decode("utf-8")
		filename =filename.replace("/","_")
		list_images=[]
		for batchs in list_batchs:
			image=np.array(batchs[i])
			if  len(image.shape)>2 and image.shape[2]==1: image=image.squeeze(2)
			list_images.append(image)
		img_visual=concatImage(list_images,offset=10)
		visualization_path = os.path.join(save_dir,filename)
		try:
			img_visual.save(visualization_path)
		except:
			print("图片保存失败【[]】".format(visualization_path))









def get_image_channel(img: Image):
    assert (img.mode == "L" or img.mode == "RGB")
    return len(img.getbands())



def get_pil_img(size, ch=1, fill=255):
    assert (ch == 1 or ch == 3)
    mode = "L" if ch == 1 else "RGB"
    return Image.new("L", size, fill)


def save_image(image,save_dir,filename):
	image = Image.fromarray(np.uint8(image)) if not isinstance(image, Image.Image) else image
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	visualization_path = os.path.join(save_dir, filename)
	image.save(visualization_path)
