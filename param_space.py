import  lyus.Frame as FM
from lyus.Frame.Exp.param import  Param,BindParam

base_param = Param()
base_param.exp_name=""
base_param.run_name=""
base_param.repeat=1


mu=[0,0.333,0.666,1]
# sigma=[0.01]+[0.035]*3
base_param.model=Param(option="OrdNetwork")
sigma=0.035
base_param.task="train"
base_param.method="method"



base_param.data_option="mvtec_bottle_data"

base_param.data=Param() # 
base_param.data.min_size= 512 #(512,512) # 384      
base_param.data.input_shape= 336 #(512,512) # 384
base_param.data.means=[0.485, 0.456, 0.406]
base_param.data.stds=[0.229, 0.224, 0.225]
base_param.data.roi_size_list= [256,384,512]#  E [256,384,512] # [256,256+128,256+256] # [256,512,512+256]
base_param.data.mv_method="msv"


base_param.ClipModel=Param(clip_name="AlphaClip", #  #BboxClip AlphaClip  VanillaClip
                            backbone_name="ViT-L/14",
                           classifier= "CosimClassfier",
                            input_shape=BindParam(base_param.data,"input_shape")
                            ,text_list=BindParam(base_param.data,"class_names"))


base_param.device="cuda"

base_param.run = Param(epoches=20)

base_param.train_dataloader=FM.get_common_dataloader_param(batch_size=1) #32
base_param.train_dataloader.num_workers=0
base_param.train_dataloader.prefetch_factor=None
base_param.train_dataloader.persistent_workers=False
base_param.train_dataloader.pin_memory=False
base_param.valid_dataloader=base_param.train_dataloader.clone()
base_param.valid_dataloader.shuffle=False


# base_param.optim = ET.get_common_SGD_optim_param("SGD")
# base_param.optim.SGD.lr = 0.1
# base_param.optim.SGD.weight_decay = 0.00003
# base_param.optim.lrReducer_name = "CosineAnnealingWarmRestarts"
# base_param.optim.CosineAnnealingWarmRestarts = ET.Param(T_0=200, T_mult=1, eta_min=1e-5, last_epoch=-1, verbose=False)

base_param.optim_option="SGD"  
# base_param.optim_option="Adam"  

base_param.optim = FM.get_common_optim_param(base_param.optim_option)
base_param.optim.SGD.lr=0.001  #0.001
base_param.optim.lrReducer_name = "CosineAnnealingLR"
base_param.optim.eta_min_ratio =  0.01
base_param.optim.CosineAnnealingLR = Param(T_max=BindParam(base_param.run,"epoches"), eta_min=1e-5, verbose=True)

base_param.hook = Param()
base_param.hook.CheckPointHook = Param(save_period=5,
                                    model_name=BindParam(base_param.model,"option"),
                                    extra_keys=[])
base_param.hook.AdjustModelModeHook=Param(finetune_epoch=5)





############### for test


base_param.run = Param(epoches=15)
base_param.hook.CheckPointHook.save_period=3
base_param.optim.SGD.lr=0.01
base_param.train_dataloader.batch_size=64



base_param.debug=Param(train_process=True,load_weight=False,filtered_module=["proto"])

# base_param.data.class_names=class_names  # 
# base_param.debug=Param(train_process=False,load_weight=False,filtered_module=["proto"])
base_param.debug=Param(k_patch=2,sdpa_scale=300,ft_epo=500,train_sk=False,  # None
                       acti_beta=20,
                       fvns=1,fvnq=1,
                       trip_loss_weight=4,trip_loss_margin=0.5, 
                       mulit_view=1,
                       k_shot=5,
                       num_sampling=5 ,
                       infer_style="assemble_uncertainty",
                       zip_config_index=6,
                       text_logits_wight=0,
                       uncertainty_alpha=2)




##################


param_space=FM.ParamSpace()
param_space.base_param=base_param

