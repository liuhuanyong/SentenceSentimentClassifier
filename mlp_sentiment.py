#!/usr/bin/env python3
# coding: utf-8
# File: mlp_sentiment.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-20
#!/usr/bin/env python3

import gensim
import numpy as np
from keras.models import load_model

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

'''三层mlp进行训练，迭代20次'''
def train_mlp(X_train, Y_train, X_test, Y_test):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    model = Sequential()
    model.add(Dense(64, input_dim=(200), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=100, epochs=20, validation_data=(X_test, Y_test))
    model.save('./model/sentiment_mlp_model.h5')
    '''
    1 [==============================] - 1s 25us/step - loss: 1.7893 - acc: 0.6168 - val_loss: 0.5576 - val_acc: 0.7076
    5 [==============================] - 0s 19us/step - loss: 0.4499 - acc: 0.7987 - val_loss: 0.4056 - val_acc: 0.8204
    10 [==============================] - 0s 17us/step - loss: 0.4043 - acc: 0.8274 - val_loss: 0.4016 - val_acc: 0.8341
    15 [==============================] - 0s 17us/step - loss: 0.3815 - acc: 0.8397 - val_loss: 0.3821 - val_acc: 0.8345
    20 [==============================] - 0s 17us/step - loss: 0.3746 - acc: 0.8432 - val_loss: 0.3842 - val_acc: 0.8359
    '''

'''实际应用，测试'''
def predict_mlp(model_filepath):
    model = load_model(model_filepath)
    sentence1 = '这个 电视 真 尼玛 垃圾 ， 老子 再也 不买 了'  # [[0.0942708 0.9058427]]
    sentence2 = '这件 衣服 真的 太 好看 了 ！ 好想 买 啊 ' # [[0.6489922  0.34993422]]
    sentence_vector1 = np.array([rep_sentencevector(sentence1)])
    sentence_vector2 = np.array([rep_sentencevector(sentence2)])
    print('test after load: ', model.predict(sentence_vector1))
    print('test after load: ', model.predict(sentence_vector2))


if __name__ == '__main__':
    #X_train, Y_train, X_test, Y_test = build_traindata()
    model_filepath = './model/sentiment_mlp_model.h5'
    #print(X_train.shape, Y_train.shape)
    #print(X_test.shape, Y_test.shape)
    #train_mlp(X_train, Y_train, X_test, Y_test)
    predict_mlp(model_filepath)


