#!/usr/bin/env python3
# coding: utf-8
# File: cnn_sentiment.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-19

import gensim
import numpy as np
from keras.models import load_model

VECTOR_DIR = './embedding/word_vector.bin'  # 词向量模型文件
model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=False)

'''基于wordvector，通过lookup table的方式找到句子的wordvector的表示'''
def rep_sentencevector(sentence):
    word_list = [word for word in sentence.split(' ')]
    max_words = 100
    embedding_dim = 200
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for index, word in enumerate(word_list):
        try:
            embedding_matrix[index] = model[word]
        except:
            pass

    return embedding_matrix


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
            Y_train.append([0, 1])
        else:
            Y_train.append([1, 0])

    for line in open('./data/test.txt'):
        line = line.strip().strip().split('\t')
        sent_vector = rep_sentencevector(line[-1])
        X_test.append(sent_vector)
        if line[0] == '1':
            Y_test.append([0, 1])
        else:
            Y_test.append([1, 0])

    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test),


'''四层CNN进行训练，迭代20次'''
def train_cnn(X_train, Y_train, X_test, Y_test):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Embedding
    from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
    #建立sequential序贯模型
    model = Sequential()
    #input_shape = (rows行, cols列, 1) 1表示颜色通道数目, rows行，对应一句话的长度, cols列表示词向量的维度
    model.add(Conv1D(64, 3, activation='relu', input_shape=(100, 200)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


    model.fit(X_train, Y_train, batch_size=100, epochs=20, validation_data=(X_test, Y_test))
    model.save('./model/sentiment_cnn_model.h5')

    '''
    1 [==============================] - 13s 664us/step - loss: 0.4868 - acc: 0.7645 - val_loss: 0.3897 - val_acc: 0.8234
    5 [==============================] - 13s 633us/step - loss: 0.2923 - acc: 0.8794 - val_loss: 0.3376 - val_acc: 0.8527
    10 [==============================] - 12s 601us/step - loss: 0.1337 - acc: 0.9482 - val_loss: 0.5124 - val_acc: 0.8284
    15 [==============================] - 13s 631us/step - loss: 0.0729 - acc: 0.9789 - val_loss: 0.8681 - val_acc: 0.8325
    20 [==============================] - 13s 632us/step - loss: 0.0484 - acc: 0.9873 - val_loss: 1.0889 - val_acc: 0.8376
    '''

'''实际应用，测试'''
def predict_cnn(model_filepath):
    model = load_model(model_filepath)
    sentence = '这个 电视 真 尼玛 垃圾 ， 老子 再也 不买 了'  # [[2.3127215e-04 0.9977249]]
    sentence = '这件 衣服 真的 太 好看 了 ！ 好想 买 啊 ' # [[0.9936581  0.00627225]]
    sentence_vector = np.array([rep_sentencevector(sentence)])
    print(sentence_vector)
    print('test after load: ', model.predict(sentence_vector))


if __name__ == '__main__':
   # X_train, Y_train, X_test, Y_test = build_traindata()
    model_filepath = './model/sentiment_cnn_model.h5'
   # print(X_train.shape, Y_train.shape)
   # print(X_test.shape, Y_test.shape)
   # train_cnn(X_train, Y_train, X_test, Y_test)
    predict_cnn(model_filepath)


