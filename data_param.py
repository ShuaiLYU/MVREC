import  lyus.Frame as FM
from lyus.Frame import  Param,BindParam

def set_data_param(data_option):



    MVTec_FS_Root="/media/lyushuai/WiseEye/PublicDataset/MVTec_AD/MVTec_FS/"
    ###################################mvtec fs 
    mvtec_carpet_data = Param(root=MVTec_FS_Root+"image/",
                            config_name=MVTec_FS_Root+"CONFIG/"+"carpet_config1/",
                            class_names=['color', 'cut', 'hole', 'metal_contamination', 'thread'],
                            num_classes=5,
                            base_class_names=[],
                            novel_class_names=[], data_name="mvtec_carpet_data")

    mvtec_grid_data = Param(root=MVTec_FS_Root+"image/",
                            config_name=MVTec_FS_Root+"CONFIG/"+"grid_config1/",
                            class_names=['bent', 'broken', 'glue', 'metal_contamination', 'thread'],
                            num_classes=5,
                            base_class_names=[],
                            novel_class_names=[], data_name="mvtec_grid_data")

    mvtec_leather_data = Param(root=MVTec_FS_Root+"image/",
                            config_name=MVTec_FS_Root+"CONFIG/"+"leather_config1/",
                            class_names=['color', 'cut', 'fold', 'glue', 'poke'],
                            num_classes=5,
                            base_class_names=[],
                            novel_class_names=[], data_name="mvtec_leather")

    mvtec_tile_data = Param(root=MVTec_FS_Root+"image/",
                            config_name=MVTec_FS_Root+"CONFIG/"+"tile_config1/",
                            class_names=['crack', 'glue_strip', 'gray_stroke', 'oil', 'rough'],
                            num_classes=5,
                            base_class_names=[],
                            novel_class_names=[], data_name="mvtec_tile_data")

    mvtec_wood_data = Param(root=MVTec_FS_Root+"image/",
                            config_name=MVTec_FS_Root+"CONFIG/"+"wood_config1/",
                            class_names=['color',  'hole', 'liquid', 'scratch'], # 'combined',
                            num_classes=5-1,
                            base_class_names=[],
                            novel_class_names=[], data_name="mvtec_wood_data")
    
    mvtec_bottle_data=Param(root=MVTec_FS_Root+"image/",
                                config_name=MVTec_FS_Root+"CONFIG/"+"bottle_config1/",
                                class_names=['broken_large', 'broken_small', 'contamination'],
                                num_classes=3,
                                base_class_names=[],
                                novel_class_names=[],data_name="mvtec_bottle_data")

    mvtec_cable_data = Param(root=MVTec_FS_Root+"image/",
                            config_name=MVTec_FS_Root+"CONFIG/"+"cable_config1/",
                            class_names=['poke_insulation', 'bent_wire', 'missing_cable', 'cable_swap',  'cut_inner_insulation', 'missing_wire', 'cut_outer_insulation'], #'combined',
                            num_classes=8-1,
                            base_class_names=[],
                            novel_class_names=[], data_name="mvtec_cable_data")
    mvtec_capsule_data = Param(root=MVTec_FS_Root+"image/",
                            config_name=MVTec_FS_Root+"CONFIG/"+"capsule_config1/",
                            class_names=['squeeze', 'crack', 'faulty_imprint', 'poke', 'scratch'],
                            num_classes=5,
                            base_class_names=[],
                            novel_class_names=[], data_name="mvtec_capsule_data")
    mvtec_hazelnut_data = Param(root=MVTec_FS_Root+"image/",
                                config_name=MVTec_FS_Root+"CONFIG/"+"hazelnut_config1/",
                                class_names=['crack', 'cut', 'hole', 'print'],
                                num_classes=4,
                                base_class_names=[],
                                novel_class_names=[], data_name="mvtec_hazelnut_data")
    mvtec_metal_nut_data = Param(root=MVTec_FS_Root+"image/",
                                config_name=MVTec_FS_Root+"CONFIG/"+"metal_nut_config1/",
                                class_names=['bent', 'color', 'flip', 'scratch'],
                                num_classes=4,
                                base_class_names=[],
                                novel_class_names=[], data_name="mvtec_metal_nut_data")
    mvtec_pill_data = Param(root=MVTec_FS_Root+"image/",
                            config_name=MVTec_FS_Root+"CONFIG/"+"pill_config1/",
                            class_names=[ 'color', 'crack', 'faulty_imprint', 'pill_type', 'contamination', 'scratch'], # 'combined',
                            num_classes=7-1,
                            base_class_names=[],
                            novel_class_names=[], data_name="mvtec_pill")
    mvtec_screw_data = Param(root=MVTec_FS_Root+"image/",
                            config_name=MVTec_FS_Root+"CONFIG/"+"screw_config1/",
                            class_names=['manipulated_front', 'scratch_head', 'scratch_neck', 'thread_side', 'thread_top'],
                            num_classes=5,
                            base_class_names=[],
                            novel_class_names=[], data_name="mvtec_screw_data")

    mvtec_transistor_data = Param(root=MVTec_FS_Root+"image/",
                                config_name=MVTec_FS_Root+"CONFIG/"+"transistor_config1/",
                                class_names=['bent_lead', 'cut_lead', 'damaged_case', 'misplaced'],
                                num_classes=4,
                                base_class_names=[],
                                novel_class_names=[], data_name="mvtec_transistor_data")
    mvtec_zipper_data = Param(root=MVTec_FS_Root+"image/",
                            config_name=MVTec_FS_Root+"CONFIG/"+"zipper_config1/",
                            class_names=[ 'broken_teeth', 'split_teeth', 'rough', 'squeezed_teeth', 'fabric_border', 'fabric_interior'], # 'combined',
                            num_classes=7-1,
                            base_class_names=[],
                            novel_class_names=[], data_name="mvtec_zipper_data")

    # mvtec_toothbrush_data = Param(root=MVTec_FS_Root+"image/",
    #                             config_name=MVTec_FS_Root+"CONFIG/"+"toothbrush_config1/",
    #                             class_names=['defective'],
    #                             num_classes=1,
    #                             base_class_names=[],
    #                             novel_class_names=[], data_name="mvtec_toothbrush_data")



    return eval(data_option)