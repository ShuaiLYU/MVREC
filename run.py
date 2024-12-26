import torch
import os.path
import sys
sys.path.append("./")
sys.path.append("../")
ganrao='/home/lyushuai/Projects/wise_pro/FabricDataSolution'
if ganrao in sys.path:
    sys.path.remove('/home/lyushuai/Projects/wise_pro/FabricDataSolution')
import lyus
print(lyus)

import  lyus.Frame as FM
from modules import *
from lyus.Frame.eval_tool import EvalTool
from fewshot_process import FewShotEvalFunc



def build_global_config(SAVE_ROOT=None,PROJECT_NAME=None):
    if PROJECT_NAME is None:
        PROJECT_NAME = FM.Exp.get_folder_name_of_file(__file__)
    if SAVE_ROOT is None:
        cul_dir =os.path.split(os.path.realpath(__file__))[0]
        SAVE_ROOT= os.path.join(cul_dir,"OUTPUT")
    return SAVE_ROOT,PROJECT_NAME
SAVE_ROOT, PROJECT_NAME = build_global_config()




def main_process(EXPER,return_results):
    PARAM=EXPER.get_param()


    train_data,_= create_train_data(EXPER) #create_train_data_support
    return_results["data_name"]=train_data.data_name

    model=create_model(EXPER)

    ck=CheckPointHook(**PARAM.hook.CheckPointHook)
    ck.bind_model(model)
    # if PARAM.debug.load_weight:
   
    #     if PARAM.debug.filtered_module:
    #         ck.filtered_module=PARAM.debug.filtered_module
    #     ck.load_the_last()

    evalTool=EvalTool(PARAM.device,os.path.join(EXPER.get_save_dir(),"visual"),
                    PARAM.data.means,
                    PARAM.data.stds,num_workers=0,batch_size=1).load(model)
    
    valid_data=None
    valid_data=create_valid_data(EXPER) #create_valid_data_novel

    eval_func= FewShotEvalFunc(support_data=train_data, query_data= valid_data,
                               metric=Accucary(len(PARAM.data.class_names)),
                            pred_name="predicts",label_name="y",k_shot= PARAM.debug.k_shot,
                            num_sampling= PARAM.debug.num_sampling)

    results= evalTool.run_eval(eval_func)
    for k, v in  results.items(): return_results[k]=v
    return return_results


if __name__ == '__main__':

    # torch.autograd.set_detect_anomaly(True)
    # import multiprocessing

    # if multiprocessing.get_start_method(allow_none=True) is None:
    #     torch.multiprocessing.set_start_method("spawn")
    from param_space import  param_space
    from data_param import set_data_param
    from lyus.Frame import CsvLogger
    arg_map= param_space.parse_args()

    run_name=arg_map["run_name"]
    exp_name=arg_map["exp_name"]
    cols= CsvLogger(os.path.join(SAVE_ROOT,PROJECT_NAME,exp_name),"results").get_cols("run_name")

    if any([  run_name==c for c in cols]):
        
        print(f"experiment for {run_name} have been run！")

    else:
        param_group=param_space.to_group()
        for idx,(PARAM_NAME,PARAM) in enumerate(param_group.items()):
            data=  set_data_param(PARAM.data_option).__dict__

            PARAM.data.regist_from_dict(data)

            print(PARAM.data)
            EXPER=FM.build_new_exper(PARAM_NAME,PARAM,SAVE_ROOT, PROJECT_NAME,exp_name=exp_name)
                    # 打印 PROJECT_NAME 检查结果
            print("PROJECT_NAME:", PROJECT_NAME)
            print("SAVE_ROOT:", SAVE_ROOT)


            model=create_model(EXPER)
            results={"param_name":PARAM_NAME,"run_name":PARAM.run_name  } # PARAM
            results= main_process(EXPER,results)
            results["hyper_param"]=PARAM.debug.to_dict()
    

            CsvLogger(os.path.join(SAVE_ROOT,PROJECT_NAME,PARAM.exp_name),"results").append_one_row(results)

        
        csv=CsvLogger(os.path.join(SAVE_ROOT,PROJECT_NAME,exp_name),"results")
        mv_average_results=csv.get_average_results(key_name="data_name",val_name="mv_acc",
                                                condition={"run_name": run_name},add_keys=["hyper_param"])
        print(mv_average_results)


        wovm_average_results=csv.get_average_results(key_name="data_name",val_name="womv_acc",
                                                condition={"run_name": run_name},add_keys=["hyper_param"])

        mv_average_results["mv_infer"]=1
        wovm_average_results["mv_infer"]=0

        print(wovm_average_results)
        CsvLogger(os.path.join(SAVE_ROOT,PROJECT_NAME,exp_name),"average_results").append_one_row(mv_average_results,strict=False)
        CsvLogger(os.path.join(SAVE_ROOT,PROJECT_NAME,exp_name),"average_results").append_one_row(wovm_average_results,strict=False)
