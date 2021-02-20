""" 训练文件
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2021/2/3
"""
import torch
import os
from dataset.vocab.vocab import Vocab
from dataset.dataset import MyDataSet
from dataset.dataloader import MyDataLoader
from transformers import AlbertForSequenceClassification
from transformers import AlbertTokenizer
from transformers import AlbertConfig
from seed_all import seed_all
import logging


if __name__ == '__main__':
    # 设置随机种子
    seed_all(2)
    data_dir = 'dataset/SST-1'
    train_data_path = os.path.join(data_dir, 'train.tsv')
    dev_data_path = os.path.join(data_dir, 'dev.tsv')
    test_data_path = os.path.join(data_dir, 'test.tsv')
    vocab_path = os.path.join(data_dir, 'sst.vocab')
    save_path = 'output/AlBert.pkl'
    log_path = 'output/albert_log.log'

    BATCH_SIZE = 32
    max_length = 56
    embedding_size = 256
    hidden_size = 1024
    lr = 0.0003
    output_per_batchs = 10
    test_per_batchs = 60
    test_batchs = 30
    groups = 3
    p=0.1
    # 加载字典
    vocab = AlbertTokenizer(
        vocab_file=vocab_path
    )
    # 创建数据集
    train_data_set = MyDataSet(train_data_path, vocab, max_length)
    test_data_set = MyDataSet(test_data_path, vocab, max_length)
    # 创建加载器
    train_data_loader = MyDataLoader(train_data_set, shuffle=True, batch_size=BATCH_SIZE)
    test_data_loader = MyDataLoader(test_data_set, shuffle=True, batch_size=BATCH_SIZE)
    # 是否用GPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # 日志记录器
    logging.basicConfig(level=logging.DEBUG, # 控制台打印的日志级别
                        filename=log_path,
                        filemode='a',        # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                                         # a是追加模式，默认如果不写的话，就是追加模式
                        format='%(message)s' # 日志格式
                    )
    file_handle = logging.FileHandler(log_path, encoding = 'utf-8')
    logger = logging.getLogger('transformer')
    logger.addHandler(file_handle)
    # 模型初始化
    config = AlbertConfig(
        vocab_size=12000,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        num_hidden_layers=12,
        num_hidden_groups=1,
        num_attention_heads=64,
        intermediate_size=4096
    )
    model = AlbertForSequenceClassification(
        config
    )
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=1e-5)
    # 开始训练
    for i in range(100):
        print('='*8 + '开始训练' + '='*8)
        logger.info('='*8 + '开始训练' + '='*8)
        model.train()
        loss_sum = 0
        for epoch, data in enumerate(train_data_loader):
            X, Y = data
            optimizer.zero_grad()
            loss = model(X, Y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()
            # 打印训练情况
            if((epoch + 1) % output_per_batchs == 0):
                print('itor: {}: epoch: {}/{}  loss: {}'.format(i + 1, epoch + 1, len(train_data_set) // 32, loss_sum / output_per_batchs))
                logger.info('itor: {}: epoch: {}/{}  loss: {}'.format(i + 1, epoch + 1, len(train_data_set) //32, loss_sum / output_per_batchs))
                loss_sum = 0
            ############################### 测试 ######################################
            if ((epoch + 1) % test_per_batchs == 0):
                print('-'*8 + '开始测试' + '-'*8)
                logger.info('-'*8 + '开始测试' + '-'*8)
                with torch.no_grad():
                    accuracy = 0
                    test_total = 0
                    model.eval()
                    is_print = True
                    for epoch, data in enumerate(test_data_loader):
                        X, Y = data
                        Y = Y.to(device=device).squeeze(dim=1)
                        y = model(X).detach()
                        if is_print:
                            print('target:', Y)
                            print('predict:', y)
                            is_print = False
                        accuracy += torch.sum(y == Y).cpu()
                        test_total += Y.shape[0]
                        if (epoch + 1) % test_batchs == 0:
                            break   
                    print('正确个数:{}, 总数:{}, 测试结果accu: {}'.format(accuracy, test_total, float(accuracy) / test_total))
                    logger.info('正确个数:{}, 总数:{}, 测试结果accu: {}'.format(accuracy, test_total, float(accuracy) / test_total))
                    torch.save(model.state_dict(), save_path)
                model.train()
######################################## 最终测试 #############################
    print('-'*8 + '开始测试' + '-'*8)
    with torch.no_grad():
        accuracy = 0
        model.eval()
        for epoch, data in enumerate(test_data_loader):
            X, Y = data
            Y = Y.to(device=device).squeeze(dim=1)
            y = model(X).detach()
            accuracy += torch.sum(y == Y).cpu()
        print('正确个数:{}, 总数:{}, 测试结果accu: {}'.format(accuracy, len(test_data_set), float(accuracy) / len(test_data_set)))
        torch.save(model.state_dict(), save_path)