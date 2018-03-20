#!/usr/bin/env python3
# coding: utf-8
# File: lstm_sentiment.py
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

'''三层lstm进行训练，迭代20次'''
def train_lstm(X_train, Y_train, X_test, Y_test):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    import numpy as np
    data_dim = 200  # 对应词向量维度
    timesteps = 100  # 对应序列长度
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=100, epochs=20, validation_data=(X_test, Y_test))
    model.save('./model/sentiment_lstm_model.h5')
    '''/
    1 [==============================] - 41s 2ms/step - loss: 0.5384 - acc: 0.7142 - val_loss: 0.4223 - val_acc: 0.8281
    5 [==============================] - 38s 2ms/step - loss: 0.2885 - acc: 0.8904 - val_loss: 0.3618 - val_acc: 0.8531
    10 [==============================] - 40s 2ms/step - loss: 0.1965 - acc: 0.9357 - val_loss: 0.3815 - val_acc: 0.8515
    15 [==============================] - 39s 2ms/step - loss: 0.1420 - acc: 0.9577 - val_loss: 0.5172 - val_acc: 0.8501
    20 [==============================] - 37s 2ms/step - loss: 0.1055 - acc: 0.9729 - val_loss: 0.5309 - val_acc: 0.8505
    '''

'''实际应用，测试'''
def predict_lstm(model_filepath):
    model = load_model(model_filepath)
    sentence = '这个 电视 真 尼玛 垃圾 ， 老子 再也 不买 了'#[[0.01477097 0.98522896]]
    #sentence = '这件 衣服 真的 太 好看 了 ！ 好想 买 啊 '#[[0.9843225  0.01567744]]
    sentence_vector = np.array([rep_sentencevector(sentence)])
    print(sentence_vector)
    print('test after load: ', model.predict(sentence_vector))


if __name__ == '__main__':
   # X_train, Y_train, X_test, Y_test = build_traindata()
    model_filepath = './model/sentiment_model.h5'
   # print(X_train.shape, Y_train.shape)
   # print(X_test.shape, Y_test.shape)
   # train_lstm(X_train, Y_train, X_test, Y_test)
    predict_lstm(model_filepath)
