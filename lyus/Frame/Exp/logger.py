#     def set_logger(self,filename="log"):# 单实例模式
#         self.logger = logging.getLogger()
#         del self.logger
#         print(self.save_dir)
#         if (not os.path.exists(self.save_dir)):
#             os.makedirs(self.save_dir)
#         LOG_FORMAT = "[%(asctime)s %(name)s %(levelname)s %(pathname)s]\n %(message)s "  # 配置输出日志格式
#         DATE_FORMAT = '%Y-%m-%d  %H:%M:%S %a '  # 配置输出时间的格式，注意月份和天数不要搞乱了
#         logging.basicConfig(level=logging.INFO,
#                             format=LOG_FORMAT,
#                             datefmt=DATE_FORMAT,
#                             filename=os.path.join(self.save_dir,filename)  # 有了filename参数就不会直接输出显示到控制台，而是直接写入文件
#                             )
        

#         self.logger = logging.getLogger()
import os,time,logging
def get_logger(save_dir,filename="log.txt"):
    # log_path = log_path + "/Log"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timer = time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime())
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s]   %(asctime)s \n%(message)s')
    consoleHandel = logging.StreamHandler()
    consoleHandel.setFormatter(formatter)
    logger.addHandler((consoleHandel))
    txthandle = logging.FileHandler((save_dir + '/' +filename))
    txthandle.setFormatter(formatter)
    logger.addHandler(txthandle) 
    return logger
