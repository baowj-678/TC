""" MT-LSTM模型参数设置文件
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/30
"""
import torch


class Config(object):
    def __init__(self):
        self.ENV = 'default'            #当前的环境参数
        self.Introduce = 'Not at the moment'    #对此次实验的描述
        self.VERSION = 1                #当前版本

	############################# GPU配置 ##############################
        self.GPU = dict(
            use_gpu = True,             #是否使用GPU，True表示使用
            device_id = [0],            #所使用的GPU设备号，type=list
        )


        self.CONFIG = dict(
            dataset_name = 'SST-1',         #数据集的名称
            trainer_name = 'TextClassifyTrainer',     # 训练器的名称
            criterion_name = 'BCELoss',       #损失函数的名称
            optimizer_name = 'Adam',     #优化器的名称
            lr = 3e-3,                      # 学习率
            adjust_lr = True,               #是否自动的变化学习率
            load_model = False,              #是否加载预训练模型（测试、迁移）
        )

        ########################## 训练参数设置 ##########################
        self.ARG = dict(
            epoch = 20,         #训练epoch
            batch_size = 8,    #训练集batch_size
        )

        #------------------------------------------------损失函数选择
        self.FocalLoss = dict(
            alpha = [0.25, 0.75],
            gamma = 2,
            num_classes = 2,
            size_average = True,
        )
        self.CrossEntropyLoss = dict(
            weight = torch.tensor([1, 1.25]).float().cuda(),
        )
        self.My_MSE_loss = dict(
            scale = 5,
        )
        self.BCELoss = dict(

        )
        
        
        #------------------------------------------------网络模型
        self.TextClassify = dict(
            version = 'baseline', 
            model_name = 'bert-base-chinese',       ## 模型采用的预训练模型      
            model_path = "C:/Users/WILL/NLP/BERT/bert-base-chinese",   ## bert预训练模型所在的文件夹
            output_dim = 1,         # 模型的输出 
        )


        #------------------------------------------------优化器
        self.Adam = dict(
            lr = 2e-5,                  #学习率
            weight_decay = 5e-4,        #权重衰减
        )


        
        self.WaiMai = dict(
            dirname = '/home/baiding/Desktop/Study/NLP/dataset/waimaiClassify',            #外卖情感分类数据集存放的文件夹
            prop = 0.9,         #训练集所占的比例
            model_name = 'bert-base-chinese',       #模型采用的分词策略
            len_seq = 512,                  #一个序列的长度
        )
        self.QAMatch = dict(
            dirname = 'D:/NLP/competition/dataset',            #外卖情感分类数据集存放的文件夹
            prop = 0.9,         #训练集所占的比例
            model_name = 'bert-base-chinese',       #模型采用的分词策略
            len_seq = 300,                  #一个序列的长度
            small_test = False,
        )

        
        ################################ 学习率变化 ###################
        self.LrAdjust = dict(
            lr_step = 1,                   #学习率变化的间隔
            lr_decay = 0.5,                 #学习率变化的幅度
            increase_bottom = 0,            #退火前学习率增加的上界
            increase_amp = 1.1,             #学习率增加的幅度
        )


        ########################### 模型加载 ###############################
        self.LoadModel = dict(
            filename = 'C:/Users/WILL/NLP/BERT/bert-base-chinese/pytorch_model.bin',     #加载模型的位置，与上面模型要对应
        )


        ########################### checkpoint ############################
        self.Checkpoint = dict(
            checkpoint_dir = './checkpoint/{}_{}_{}_V{}'.format(
                self.CONFIG['dataset_name'], self.CONFIG['model_name'],
                getattr(self, self.CONFIG['model_name'])['version'],
                self.VERSION),                          # checkpoint 所在的文件夹
            checkpoint_file_format = self.CONFIG['model_name']+'_Epoch{}.pkl',     #模型文件名称格式，分别表示模型名称、Epoch
            model_best = 'model_best.ptk',            #最好的模型名称，暂时未用到
            log_file = 'log_{}.log',                         #log文件名称，要加时间
            save_period = 1,                            #模型的存储间隔
        )


    def log_output(self):
        log = {}
        log['ENV'] = self.ENV
        log['Introduce'] = self.Introduce
        log['CONFIG'] = self.CONFIG
        for name,value in self.CONFIG.items():
            if type(value) is str and hasattr(self,value):
                log[value] = getattr(self,value)
            else:
                log[name] = value
        for name,value in self.ARG.items():
            log[name] = value
        return log

