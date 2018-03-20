#!/usr/bin/env python3
# coding: utf-8
# File: knn_sentiment.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-20

import gensim
import numpy as np
from sklearn.externals import joblib

VECTOR_DIR = './embedding/word_vector.bin'  # 词向量模型文件
model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=False)

'''基于wordvector，通过lookup table的方式找到句子的wordvector的表示'''
def rep_sentencevector(sentence):
    '''通过向量求和的方式标识sentence vector'''
    word_list = [word for word in sentence.split(' ')]
    embedding_dim = 200
    embedding_matrix = np.zeros(embedding_dim)
    for index, word in enumerate(word_list):
        try:
            embedding_matrix += model[word]
        except:
            pass

    return embedding_matrix/len(word_list)

'''构造训练数据'''
def build_traindata():
    X_train = list()
    Y_train = list()
    X_test = list()
    Y_test = list()
    for line in open('./data/train.txt'):
        line = line.strip().strip().split('\t')
        sent_vector = rep_sentencevector(line[-1])

        X_train.append(sent_vector)
        if line[0] == '1':
            Y_train.append(1)
        else:
            Y_train.append(0)

    for line in open('./data/test.txt'):
        line = line.strip().strip().split('\t')
        sent_vector = rep_sentencevector(line[-1])
        X_test.append(sent_vector)
        if line[0] == '1':
            Y_test.append(1)
        else:
            Y_test.append(0)

    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test),

'''基于knn分类器算法'''
def train_knn(X_train, Y_train, X_test, Y_test):
    from sklearn.neighbors import KNeighborsClassifier
    '''
    for x in range(1, 15):
        model = KNeighborsClassifier(n_neighbors=x)
        model.fit(X_train, Y_train)
        preds = knnclf.predict(X_test)
        num = 0
        num = 0
        preds = preds.tolist()
        for i, pred in enumerate(preds):
            if int(pred) == int(Y_test[i]):
                num += 1
        print('K= ' + str(x) + ', precision_score:' + str(float(num) / len(preds)))

    *****************result****************
    K= 1, precision_score:0.7169056352117372
    K= 2, precision_score:0.7189063021007003
    K= 3, precision_score:0.7600866955651884
    K= 4, precision_score:0.7519173057685895
    K= 5, precision_score:0.764754918306102
    K= 6, precision_score:0.7709236412137379
    K= 7, precision_score:0.7724241413804601
    K= 8, precision_score:0.7784261420473492
    K= 9, precision_score:0.7804268089363121
    K= 10, precision_score:0.7814271423807936
    K= 11, precision_score:0.7829276425475158
    K= 12, precision_score:0.7869289763254418
    K= 13, precision_score:0.7829276425475158
    K= 14, precision_score:0.7909303101033678
    '''
    #选择K=20进行KNN训练
    model = KNeighborsClassifier(n_neighbors=14)
    model.fit(X_train, Y_train)
    joblib.dump(model, './model/sentiment_knn_model.m')

'''基于knn分类器的预测'''
def evaluate_knn(model_filepath, X_test, Y_test):
    model = joblib.load(model_filepath)
    Y_predict = list()
    Y_test = list(Y_test)
    right = 0
    for sent in X_test:
        Y_predict.append(model.predict(sent.reshape(1, -1)))
    for index in range(len(Y_predict)):
        if Y_predict[index] == Y_test[index]:
            right += 1
    score = right / len(Y_predict)
    print('model accuray is :{0}'.format(score))#0.7909303101033678
    return score

'''实际应用测试'''
def predict_knn(model_filepath):
    model = joblib.load(model_filepath)
    sentence1 = '这个 电视 真 尼玛 垃圾 ， 老子 再也 不买 了'
    sentence2 = '这件 衣服 真的 太 好看 了 ！ 好想 买 啊 '
    rep_sen1 = np.array(rep_sentencevector(sentence1)).reshape(1, -1)
    rep_sen2 = np.array(rep_sentencevector(sentence2)).reshape(1, -1)
    print('sentence1', model.predict(rep_sen1)) #[1]
    print('sentence2', model.predict(rep_sen2)) #[0]

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = build_traindata()
    model_filepath = './model/sentiment_knn_model.m'
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    train_knn(X_train, Y_train, X_test, Y_test)
    evaluate_knn(model_filepath, X_test, Y_test)
    predict_knn(model_filepath)