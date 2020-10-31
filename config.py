""" 全局的配置文件
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/31
"""


class Config(object):
    def __init__(self):
        self.ENV = 'default'                    #当前的环境参数
        self.Introduce = 'Not at the moment'    #对此次实验的描述
        self.VERSION = 1                        #当前版本
        ############################# GPU配置 ##############################
        self.GPU = dict(
            use_gpu=True,          # 是否使用GPU，True表示使用
        )
        ############################ 数据集配置 ############################
        self.DATA_CONFIG = dict(
            dataset_name='SST-1',  # 数据集的名称
        )

        ########################### 模型配置 ###############################
        self.CONFIG = dict(
            model_name='MT-LSTM',
            trainer_name='TextClassifyTrainer',  # 训练器的名称
            criterion_name='BCELoss',  # 损失函数的名称
            adjust_lr=True,  # 是否自动的变化学习率
            load_model=False,  # 是否加载预训练模型（测试、迁移）
        )

        ########################## 训练参数配置 ##########################
        self.ARG = dict(
            epoch=20,  # 训练epoch
            batch_size=8,  # 训练集batch_size
            lr=3e-3,  # 学习率
            optimizer_name='Adam',  # 优化器的名称
        )

        ################################ 学习率变化 ###################
        self.LrAdjust = dict(
            lr_step=1,  # 学习率变化的间隔
            lr_decay=0.5,  # 学习率变化的幅度
            increase_bottom=0,  # 退火前学习率增加的上界
            increase_amp=1.1,  # 学习率增加的幅度
        )

        ########################### 模型保存/加载 ###############################
        self.LoadModel = dict(
            filename='C:/Users/WILL/NLP/BERT/bert-base-chinese/pytorch_model.bin',  # 加载模型的位置，与上面模型要对应
        )

        ########################### checkpoint ############################
        self.Checkpoint = dict(
            checkpoint_dir='./checkpoint/{}_{}_{}_V{}'.format(
                self.CONFIG['dataset_name'], self.CONFIG['model_name'],
                getattr(self, self.CONFIG['model_name'])['version'],
                self.VERSION),  # checkpoint 所在的文件夹
            checkpoint_file_format=self.CONFIG['model_name'] + '_Epoch{}.pkl',  # 模型文件名称格式，分别表示模型名称、Epoch
            model_best='model_best.ptk',  # 最好的模型名称，暂时未用到
            log_file='log_{}.log',  # log文件名称，要加时间
            save_period=1,  # 模型的存储间隔
        )


if __name__ == '__main__':
    pass