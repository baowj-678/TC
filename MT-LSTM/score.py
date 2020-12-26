""" 计算准确率、召回率和F1
"""
def getTPFN(predict, target, entity_vocab):
    """
    @param:
    :predict: (batch_size, seq_len)
    :target: (batch_size, seq_len)
    @return:
    :accuracy: 正确率
    :precision: 准确率
    :recall: 召回率
    :F1 :F值
    """
    TP = 0 # 将正类预测为正类数
    TN = 0 # 将负类预测为负类数
    FN = 0 # 将正类预测为负类数
    FP = 0 # 将负类预测为正类数
    for predict_, target_ in zip(predict, target):
        for predict_i, target_i in zip(predict_, target_):
            if predict_i == entity_vocab.o:
                if target_i == predict_i:
                    TN += 1
                else:
                    FN += 1
            else:
                if target_i == predict_i:
                    TP += 1
                else:
                    FP += 1
    return (TP, FP, TN, FN)

def getScore(predict, target, entity_vocab):
    """
    @param:
    :predict: (batch_size, seq_len)
    :target: (batch_size, seq_len)
    @return:
    :accuracy: 正确率
    :precision: 准确率
    :recall: 召回率
    :F1 :F值
    """
    TP = 0 # 将正类预测为正类数
    TN = 0 # 将负类预测为负类数
    FN = 0 # 将正类预测为负类数
    FP = 0 # 将负类预测为正类数
    for predict_, target_ in zip(predict, target):
        for predict_i, target_i in zip(predict_, target_):
            if predict_i == entity_vocab.o:
                if target_i == entity_vocab.o:
                    TN += 1
                else:
                    FN += 1
            else:
                if target_i == entity_vocab.o:
                    FP += 1
                else:
                    TP += 1
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    return (accuracy, precision, recall, F1)

    
