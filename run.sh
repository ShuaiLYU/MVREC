
mv_method=mso
clip_name=AlphaClip
datasets="mvtec_carpet_data mvtec_grid_data mvtec_leather_data mvtec_tile_data mvtec_wood_data mvtec_bottle_data mvtec_cable_data mvtec_capsule_data mvtec_hazelnut_data mvtec_metal_nut_data mvtec_pill_data mvtec_screw_data mvtec_transistor_data mvtec_zipper_data"

for k_shot in 1 3 5
do
    exp_name=best_exp_notrain
    acti_beta=32
    classifier=EchoClassfier  # zip-adpater
      python run.py --data_option $datasets \
      --ClipModel.classifier $classifier \
      --ClipModel.backbone_name ViT-L/14 \
      --ClipModel.clip_name   $clip_name \
      --debug.k_shot $k_shot  \
      --data.input_shape 224   \
      --data.mv_method $mv_method \
      --debug.acti_beta $acti_beta \
      --exp_name $exp_name \
      --run_name MVRec-$mv_method-$classifier-ks$k_shot-acti$acti_beta
done



mv_method=mso
exp_name=best_exp_train
clip_name=AlphaClip
datasets="mvtec_carpet_data mvtec_grid_data mvtec_leather_data mvtec_tile_data mvtec_wood_data mvtec_bottle_data mvtec_cable_data mvtec_capsule_data mvtec_hazelnut_data mvtec_metal_nut_data mvtec_pill_data mvtec_screw_data mvtec_transistor_data mvtec_zipper_data"


for k_shot in 1 3 5
do
    acti_beta=1
    classifier=EchoClassfierF   # zip-adpater-F
      python run.py --data_option $datasets \
      --ClipModel.classifier $classifier \
      --ClipModel.backbone_name ViT-L/14 \
      --ClipModel.clip_name   $clip_name \
      --debug.k_shot $k_shot  \
      --data.input_shape 224   \
      --data.mv_method $mv_method \
      --debug.acti_beta $acti_beta \
      --exp_name $exp_name \
      --run_name MVRec-$mv_method-$classifier-ks$k_shot-acti$acti_beta
done




