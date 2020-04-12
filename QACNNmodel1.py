# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 20:01:02 2019

@author: umd1231
"""
import random
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords as pw
import nltk.stem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

###read dataset###
data = pd.read_csv(r'ChatbotQuestionsCSV.csv')

train_data_set = data['QUESTION'].values
train_labels = data['LABEL'].values 

#print('data_set',data['QUESTION'].values)
#print('labels',data['LABEL'].values)

###prepare data###
all_data = [] 
#cacheStopWords=pw.words("english")

for train_data in train_data_set:    
    train_data = train_data.lower()
    #train_data=''.join([word+" " for word in train_data.split() if word not in cacheStopWords])
    #str_data = nltk.word_tokenize(train_data)
    #s = nltk.stem.SnowballStemmer('english')
    all_data.append(train_data)
#print(all_datas)
    
###build vocabulary###
dictionary = {}

for data in all_data:
    for word in data.split():   
        if word not in dictionary:  
            dictionary[word] = len(dictionary) + 1
#print(dictionary)

###embeding train data###
max_length = 10
input_data = np.zeros(shape = (len(all_data), max(dictionary.values()) + 1, max_length))
# print('shape',input_data.shape)

for i, data in enumerate(all_data):  
    for j, word in list(enumerate(data.split()))[:max_length]: 
        index = dictionary.get(word)
        input_data[i, index, j] = 1
#print(input_data)
#input_data=pad_sequences(input_all_data,maxlen=50)

#makes maxlength flexible#
classes = max(train_labels) + 1
# print(classes)
#creates array with dimesions :length of train_labels , total amount of response classes
one_hot_label = np.zeros(shape = (train_labels.shape[0], classes))
one_hot_label[np.arange(0, train_labels.shape[0]), train_labels] = 1
input_label = one_hot_label
#print('input_label',input_label)
zippedDataAndLabels=list(zip(input_data,input_label))
# print(zippedDataAndLabels)
random.shuffle(zippedDataAndLabels)

input_data,input_label=zip(*zippedDataAndLabels)
input_data=np.array(input_data)
input_label=np.array(input_label)
print(input_data)
print(input_label)
###split train set and validation set###
train_size = 200
x_train = input_data[:train_size]
x_val = input_data[train_size:]
y_train = input_label[:train_size]
y_val = input_label[train_size:]

#max_min=preprocessing.MinMaxScaler()
#x_train = max_min.fit_transform(x_train)
#x_val = max_min.fit_transform(x_val)

#x_train = np.reshape(x_train,(18,50,1))
#x_val = np.reshape(x_val,(9,50,1))


###CNN model###
model = Sequential()
model.add(Convolution1D(256, 3, padding = 'same', input_shape = (len(dictionary) + 1, max_length)))
model.add(MaxPool1D(3, 3, padding = 'same'))
model.add(Convolution1D(128, 3, padding = 'same'))
model.add(MaxPool1D(3, 3, padding = 'same'))
model.add(Convolution1D(64, 3, padding = 'same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(BatchNormalization())  
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(classes, activation = 'softmax'))
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


history = model.fit(x_train,
          y_train,
          batch_size = train_size,
          epochs = 300,
          validation_data = (x_val, y_val))

#y_predict = model.predict_classes(x_val)
# print(y_predict)

###save model###
model.save('QACNNmodel1.h5')