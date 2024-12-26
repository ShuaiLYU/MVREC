"""

20221028


"""
import copy,sys
from collections  import  OrderedDict
__all__=["get_common_optim_param","Param","BindParam","ParamSpace","get_single_gpu_device_param","get_common_dataloader_param"]

from .utils import  is_scalar,check_scalar
from .arg import ArgTool

class BindParam(object):
    
    def __init__(self, param, key):
        self.param = param
        self.key = key

    def get(self):
        return self.param[self.key]
    
    def set(self, value):
        self.param[self.key] = value
    def __str__(self):
        return f"({self.get()})BindParam({self.param.param_name}.{self.key}))"
    
class Param(object):
    def __init__(self, param_name="param", **kargs):
        self._param_name = param_name
        self.regist_from_dict(kargs)


    def regist_from_dict(self, _dict):
        assert isinstance(_dict, dict)
        for key, val in _dict.items():
            self.check_key(key)
            self.set(key, val)
        return self

    def regist_child(self, param_name: str, init_param=None):
        self[param_name] = init_param.clone() if init_param is not None else Param()
        return self[param_name]

    @property
    def name(self):
        name = self._param_name.split(".")[-1]
        return name

    @property
    def param_name(self):
        return self._param_name

    def check_key(self, key):
        assert (key != "param_name")
        assert (key != "name")
        assert (key != "keys")
        assert (key != "vals")
        assert (key != "items")

    def update_name(self, last_name, key):
        self._param_name = last_name + "." + key
        for key, val in self.__dict__.items():
            if isinstance(val, Param):
                val.update_name(self._param_name, key)


    # 功能 A["a"]
    def __setitem__(self, key, value):
        super(Param, self).__setattr__(key, value)
        if isinstance(value, Param):
            value.update_name(self._param_name, key)

    # self.__dict__[key] = value
    def __getitem__(self, attr):
        # print(attr)
        return self.__getattribute__(attr)


    def __delitem__(self, key):
        try:
            del self.__dict__[key]
        except KeyError as k:
            return None

    # 功能  A.a
    def __setattr__(self, key, value):
        if hasattr(self,key) and isinstance(self[key],BindParam):
            self[key].set(value)
            print("set bind param {}".format(key))
        else:
            super(Param, self).__setattr__(key, value)
            if isinstance(value, Param):
                value.update_name(self._param_name, key)

        
    def __getattribute__(self, attr):
        # print(attr)
        value=super(Param, self).__getattribute__(attr)
        if isinstance(value,BindParam):
            value=value.get()
        return value


    # 	"""|
    # 	重载此函数防止属性不存在时__getattribute__报错，而是返回None
    # 	那“_ getattribute_”与“_ getattr_”的最大差异在于：
    # 	1. 无论调用对象的什么属性，包括不存在的属性，都会首先调用“_ getattribute_”方法；
    # 	2. 只有找不到对象的属性时，才会调用“_ getattr_”方法；
    # 	:param attr:
    # 	:return:
    # 	"""
    # 	return super(Param, self).__getattr__(attr)
    # raise Exception("attr:[{}] is not existing".format(attr))
    def __delattr__(self, key):
        try:
            del self.__dict__[key]
        except KeyError as k:
            return None

    # def __str__(self):
    # 	string=""
    # 	for key,val in self.__dict__.items():
    # 		if key is "_name": continue
    # 		if isinstance(val,Param):
    # 			string += self._name + "{}=Param()\n".format(key)
    # 			string +="{}".format(val)
    # 		else:
    # 			string +=self._name+"{}={}\n".format(key,val)
    # 	return string
    def __str__(self):
        string = self._param_name + "=Param()\n"
        for key, val in self.__dict__.items():
            if key == "_param_name": continue
            if isinstance(val, Param):
                string += str(val)
            elif isinstance(val, str):
                string += self._param_name + ".{}='{}'\n".format(key, val)

            else:
                string += self._param_name + ".{}={}\n".format(key, val)
        return string

    def __len__(self):
        return len(self.keys())

    def keys(self):
        keys = [key for key in self.__dict__.keys() if key != "_param_name"]
        return keys

    def values(self):
        return [self[key] for key in self.keys()]

    def to_dict(self):
        return { kk:self[kk] for kk in self.keys()}   #  


    def items(self):
        return [item for item in self if item[0] in self.keys()]


    def get(self, k, v, separator="."):
        if separator not in k:
            return self[k]
        else:
            _param=self
            pointers= k.split(separator)
            for _k in  pointers[:-1]:
                _param=_param[_k]
                assert(isinstance(_param,Param))
            return _param[pointers[-1]]

    def set(self, k, v, separator="."):
        if separator not in k:
            self[k]=v
        else:
            _param=self
            pointers= k.split(separator)
            for _k in  pointers[:-1]:
                _param=_param[_k]
                assert(isinstance(_param,Param))
            _param[pointers[-1]]=v
    def bind(self,param,key):
        self[key]=BindParam(param,key)

    def hasattr(self,k,separator="."):
        if separator not in k:
            return hasattr(self,k)
        else:
            cul_key=k.split(separator)[0]
            if not hasattr(self,cul_key):
                return False
            else:
                child_param = self[cul_key]
                child_key = k[len(cul_key) + 1:]
                assert (isinstance(child_param, Param))
                return child_param.hasattr(child_key)

    def clone(self):
        return copy.deepcopy(self)


class ParamSpace(Param):

    # def get_parser(self):
    #     if not hasattr(self,"__parser "):
    #         self.__parser=lyus.ExpTool.arg.ArgumentParser()
    #         self.__parser.add_argument("-base_param", default="base", type=str)
    #     return self.__parser
    # def add_argument(self,*args,**kwargs):
    #     print(args,kwargs)
    #     self.get_parset().add_argument(*args,**kwargs)


    def parse_args(self,arg_string=None):
        if arg_string is not None:
            # argv=[ arg for arg in arg_string.strip().split(" -")  if arg != "" ]
            argv=[ arg for arg in arg_string.strip().split(" ")  if arg != "" ]
 
            args=argv[2:]
        else:
            args=sys.argv[1:]

        arg_map=OrderedDict()
        while(len(args)>0):
            residual_arg_num=len(args)
            assert (residual_arg_num>1),"wrong arg number!"
            # print(args)
            # assert False
            assert args[0].startswith("-") or  args[0].startswith("--"),"arg key must start with '-' or '--' "
            assert (not args[1].startswith("-")) , f"arg {args} value cannot start with '-' !"

            key = args[0].replace("-","")
            index = 1
            while( (index < residual_arg_num ) and (not args[index].startswith("-"))):
                index+=1
            arg_map[key]=[ check_scalar(v) for v in  args[1:index]]
            if isinstance (arg_map[key],list) and len(arg_map[key])==1: arg_map[key]=arg_map[key][0] 
            
            args=args[index:]


        self._base_param_name="base_param"
        if "base_param" in arg_map.keys():
            assert isinstance(arg_map["base_param"],list) and len(arg_map["base_param"])==1
            self._base_param_name=arg_map.pop("base_param")[0]


        # check if hasattr
        for k ,v in arg_map.items():
            base_param=self[self._base_param_name]
            assert base_param.hasattr(k), "base param do not has the attart '{}'".format(k)
            # print(k,v,type(k))
        self.arg_map =arg_map
        return self.arg_map
    

    def get_arg_combination(self,args: OrderedDict):
        keys = list(args.keys())
        param_comb = [args]
        for k in keys:
            _param_comb = []
            for param in param_comb:
                v_list = param[k]
                if not isinstance(v_list, list):
                    v_list = [v_list]
                assert (len(set(v_list)) == len(v_list)),v_list
                for v in v_list:
                    _param = copy.deepcopy(param)
                    _param[k] = v
                    _param_comb.append(_param)
            param_comb = _param_comb
        print(param_comb)
        return param_comb


    def to_group(self,arg2param=None):

        arg_comb = self.get_arg_combination(self.arg_map)

        base_param=self[self._base_param_name]
        def default_arg2param(arg,base_param):
            param = base_param.clone()
            # arg = {k: v for k, v in arg.items() if k != "base_param"}
            for k, v in arg.items():
                param.set(k, v, ".")
            return param
        if arg2param is None:arg2param=default_arg2param

        param_group={}

        for idx, arg in enumerate(arg_comb):
            arg_tool = ArgTool(base_name=self._base_param_name).init_from_arg(arg)
            exp_name = arg_tool.get_arg_name()
            param = arg2param(arg,base_param)
            param_group[exp_name]=param
            
        return   param_group



def get_common_optim_param(optimizer_name):

    assert  optimizer_name in ["SGD","Adam"]

    if optimizer_name=="SGD":
        optim_param = Param(
            optimizer_name=optimizer_name,
            lrReducer_name=None,
            with_earlyStopper=False,
            with_warmUp=False,
        )
        optim_param.SGD = Param(lr=0.03, momentum=0.9, weight_decay=0.1)  ## momentum 0.5->0.1
        optim_param.ReduceLROnPlateau = Param(mode="min", verbose=True, patience=3, factor=0.5)
        optim_param.EarlyStopper = Param(thred_loss=0.01, verbose=True, patience=10)
        optim_param.WarmUp = Param(min_lr=1e-6, num_steps=200, method='line', verbose=True)
        return  optim_param
    if optimizer_name=="Adam":
        optim_param = Param(
            optimizer_name=optimizer_name,
            lrReducer_name=None,
            with_earlyStopper=False,
            with_warmUp=False,
        )
        optim_param.Adam = Param(lr=0.001, weight_decay=0.001, betas=(0.9, 0.999))

        return  optim_param


def get_single_gpu_device_param():
    param=Param()
    param.device_name = "cuda"
    param.id = "0"
    return param


def get_common_dataloader_param(batch_size=128):
    train_loader=Param()
    train_loader.batch_size = batch_size
    train_loader.shuffle = True
    train_loader.num_workers = 8
    train_loader.sampler=None
    train_loader.batch_sampler=None
    train_loader.drop_last = False
    train_loader.persistent_workers = True
    train_loader.pin_memory = True
    train_loader.prefetch_factor = 5
    return train_loader



# def get_common_run_param():
#     param.epoches=100
#
#     return param
