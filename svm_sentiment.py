#!/usr/bin/env python3
# coding: utf-8
# File: svm_sentiment.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-20

import gensim
import numpy as np
from sklearn.externals import joblib

VECTOR_DIR = './embedding/word_vector.bin'  # 词向量模型文件
model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=False)

'''基于wordvector，通过lookup table的方式找到句子的wordvector的表示，向量求和做平均'''
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

'''基于svm分类器算法, 使用SVC算法，使用默认参数'''
def train_svm(X_train, Y_train):
    from sklearn.svm import SVC
    model = SVC(kernel='linear')
    model.fit(X_train, Y_train)
    joblib.dump(model, './model/sentiment_svm_model.m')

'''基于svm分类器的预测'''
def evaluate_svm(model_filepath, X_test, Y_test):
    model = joblib.load(model_filepath)
    Y_predict = list()
    Y_test = list(Y_test)
    right = 0
    for sent in X_test:
        Y_predict.append(model.predict(sent.reshape(1, -1))[0])
    for index in range(len(Y_predict)):
        if int(Y_predict[index]) == int(Y_test[index]):
            right += 1
    score = right / len(Y_predict)
    print('model accuray is :{0}'.format(score)) #0.8302767589196399
    return score

'''实际应用测试'''
def predict_svm(model_filepath):
    model = joblib.load(model_filepath)
    sentence1 = '这个 电视 真 尼玛 垃圾 ， 老子 再也 不买 了'
    sentence2 = '这件 衣服 真的 太 好看 了 ！ 好想 买 啊 '
    rep_sen1 = np.array(rep_sentencevector(sentence1)).reshape(1, -1)
    rep_sen2 = np.array(rep_sentencevector(sentence2)).reshape(1, -1)
    print('sentence1', model.predict(rep_sen1)) #sentence1 [1]
    print('sentence2', model.predict(rep_sen2)) #sentence2 [0]

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = build_traindata()
    model_filepath = './model/sentiment_svm_model.m'
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    train_svm(X_train, Y_train)
    evaluate_svm(model_filepath, X_test, Y_test)
    predict_svm(model_filepath)