# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 19:59:24 2019

@author: umd1231
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords as pw

data = pd.read_csv(r'ChatbotQuestionsCSV.csv')

train_questions = data['QUESTION'].values
train_labels = data['LABEL'].values 
all_questions = []
#cacheStopWords=pw.words("english")

for train_question in train_questions:
    train_question = train_question.lower()
    #train_data=''.join([word+" " for word in train_data.split() if word not in cacheStopWords])
    #str_data = nltk.word_tokenize(train_data)
    #s = nltk.stem.SnowballStemmer('english')
    all_questions.append(train_question)
#print(all_datas)
    
dictionary = {}

for all_question in all_questions:
    for word in all_question.split():
        if word not in dictionary:  
            dictionary[word] = len(dictionary) + 1

#print(dictionary)