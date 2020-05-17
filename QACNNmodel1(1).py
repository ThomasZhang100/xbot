# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 20:01:02 2019

@author: umd1231
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords as pw
import nltk.stem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from keras import layers,Input
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import random



###read dataset###
data = pd.read_csv(r'Chatbot Questions CSV - Sheet1.csv')

train_data_set = data['Question'].values
train_labels = data['Label'].values

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
max_length = 24
input_data = np.zeros(shape = (len(all_data), max(dictionary.values()) + 1, max_length)) 

for i, data in enumerate(all_data):  
    for j, word in list(enumerate(data.split()))[:max_length]: 
        index = dictionary.get(word)
        input_data[i, index, j] = 1
#print(input_data)
#input_data=pad_sequences(input_all_data,maxlen=50)

###embeding train label###
classes = max(train_labels) + 1
one_hot_label = np.zeros(shape = (train_labels.shape[0], classes))
one_hot_label[np.arange(0, train_labels.shape[0]), train_labels] = 1
input_label = one_hot_label
#print(input_label)

###split train set and validation set###
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

(input_data_shuffled,input_label_shuffled)=shuffle_in_unison(input_data,input_label)
cutoff=50
x_train = input_data_shuffled[cutoff:]
x_val = input_data_shuffled[:cutoff]
y_train = input_label_shuffled[cutoff:]
y_val = input_label_shuffled[:cutoff]

#max_min=preprocessing.MinMaxScaler()
#x_train = max_min.fit_transform(x_train)
#x_val = max_min.fit_transform(x_val)

#x_train = np.reshape(x_train,(18,50,1))
#x_val = np.reshape(x_val,(9,50,1))
print(type(x_val))
print(type(y_val))

###CNN model###
'''
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
model.add(Dense(num_labels, activation = 'softmax'))

'''
#my_input=Input(shape=(len(dictionary)+1,max_length))
def get_piece_layer():
    double_layer=Sequential()
    double_layer.add(Convolution1D(32,3,padding="same",input_shape=(len(dictionary)+1,32)))
    double_layer.add(layers.ReLU())
    double_layer.add(Convolution1D(32,3,padding="same",input_shape=(len(dictionary)+1,32)))


    second_double_layer=Sequential()
    second_double_layer.add(Convolution1D(32,3,padding="same",input_shape=(len(dictionary)+1,32)))
    second_double_layer.add(layers.ReLU())
    second_double_layer.add(Convolution1D(32,3,padding="same",input_shape=(len(dictionary)+1,32)))
    #double_layer.summary()

    starting_double=Input(shape=(len(dictionary)+1,32))
    after_double=double_layer(starting_double)

    temp=layers.Add()([starting_double,after_double])
    temp=layers.BatchNormalization()(temp)
    temp=layers.Dropout(0.2)(temp)
    
    last_double=second_double_layer(temp)
    added=layers.Add()([last_double,starting_double])
    final=layers.BatchNormalization()(added)
    final=layers.Dropout(0.2)(final)
    piece_layer=Model(inputs=starting_double,outputs=added)
    return piece_layer

    
#piece_layer.summary()
#make model
model=Sequential()
model.add(Convolution1D(32, 3, padding = 'same', input_shape = (len(dictionary) + 1, max_length)))
for i in range(30):
    model.add(get_piece_layer())
#model.add(Convolution1D(32, 3, padding = 'same'))

print(model.output_shape)
model.add(Convolution1D(64, 3, padding = 'same',strides=2,activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
model.add(Convolution1D(64, 3, padding = 'same',strides=2,activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
model.add(Convolution1D(64, 3, padding = 'same',strides=2,activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
print(model.output_shape)
model.add(Flatten())
model.add(Dense(classes,activation="softmax"))

model.summary()

model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


history = model.fit(x_train,
          y_train,
          batch_size = 10,
          epochs = 7,
          validation_data = [x_val, y_val])

y_predict = model.predict_classes(x_val)
print(y_predict)

###save model###
model.save('QACNNmodel1(1).h5')
