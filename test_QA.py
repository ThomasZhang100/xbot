# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:46:00 2019

@author: umd1231
"""

import pandas as pd
import numpy as np
import nltk
from keras.models import load_model
import speech_recognition as sr
import vocabulary1
import win32com.client as wincl
from playsound import playsound
import csv
import random
import datetime

###read answers dataset###
data = pd.read_csv(r'ChatbotAnswersTXT.txt',sep = ":")
Answers = data['ANSWER'].values
IDs = data['ID'].values


###build answers dictionary###
Answer_set = {}
Answer_set_lengths = []
oldwholeID=0
count=0
# print(IDs)

#creates
for i in range(len(Answers)):
    wholeID=int(IDs[i])
    # print(wholeID)
    if wholeID==oldwholeID:
        count+=1
    else:
        Answer_set_lengths.append(count)
        count=1
    Answer_set[IDs[i]] = Answers[i]
    oldwholeID=wholeID
# print(Answer_set_lengths)


###speak welcome###
speak = wincl.Dispatch("SAPI.SpVoice")
speak.Speak("Hello, how can I help you")

###speech recognition###
#sample_rate = 48000
#chunk_size = 2048
#r.dynamic_energy_threshold = True
#r.dynamic_energy_adjustment_damping = 0.15
#r.dynamic_energy_adjustment_ratio = 1.2
while True:
    recog_value = 0
    recog = []
    r = sr.Recognizer()
    #speak.Speak("I'm listening")
    with sr.Microphone() as source:
#        r.adjust_for_ambient_noise(source)
        playsound('ding.mp3')
        print("I am listening...")
#        audio = r.listen(source, timeout = 5, phrase_time_limit = 3)
        audio = r.listen(source)
        
        ###recognize the voice###
        try:
            recog = r.recognize_google(audio, language = 'en_US')
            print("You said: " + recog)
            recog_value = 1
            
        ###fail to recognition### 
        except sr.UnknownValueError:
            s_a = ["Sorry, please say it again after the beep"]
            print(''.join(s_a))
            speak.Speak(s_a)
    
    ###predict answer from question###
    if recog_value == 1:

        # ###quit loop###
        # if recog == "goodbye":
        #     gb=["Goodbye, thank you"]
        #     print('Answer:' + ''.join(gb))
        #     speak.Speak(gb)
        #     break
        #
        #else:
        test_data = []
        test_data.append(recog)
            
            ###load dictionary###
        dic = vocabulary1.dictionary
        # print(dic)
            ###one hot embeding###
            ###creates 3d numpyarray with zeros using three dimensions###
        max_length = 10
        x_test = np.zeros(shape=(len(test_data),
                                max(dic.values()) + 1,
                                max_length))

            #fills in empty array with true values where the words match the test data words#
        for i, test_data in enumerate(test_data):
            for j, word in list(enumerate(test_data.split()))[:max_length]:
                index = dic.get(word)
                x_test[i,index,j] = 1
            ###load the model###      
        model=load_model('QACNNmodel1.h5')
            
            ###predict class###
        y_predict = model.predict_classes(x_test)
            #print(y_predict)
            
            ###probability of prediction###
        y_probability = model.predict_proba(x_test)
            #print(y_probability)
        print(y_predict)
            ###quit the loop###
            #if y_predict == 7:
             #   y_predict7 = float(y_predict)
              #  number7 = random.randint(1,2)
              #  r_n7 = number7/10
              #  y_pre7 = y_predict7 + r_n7
              #  Final_answer7 = Answer_set[y_pre7]
              #  print('Answer:' + ''.join(Final_answer7))
               # speak.Speak(Final_answer7)
               # break

            ###current date### 
        if y_predict == 32:
            current_date = datetime.datetime.now()
            date_answer = current_date.strftime('%Y/%m/%d')
            print('Answer:' + 'Today is ' + ''.join(date_answer))
            speak.Speak('Today is ' + date_answer)
            
            ###current time###
        elif y_predict == 33:
            current_time = datetime.datetime.now()
            time_answer = current_time.strftime('%H:%M')
            print('Answer:' + 'It is currently ' + ''.join(time_answer))
            speak.Speak('It is currently' + time_answer)
                
        else:
                ###save unknow questions###
            isBreak=False
            for y_pro in y_probability:
                new_train = []
                if all(y_pro < 0.5):
                    unknown = ["Sorry, I didn't understand"]
                    print(''.join(unknown))
                    speak.Speak(unknown)
                        
                    new_train = [test_data]
                    file=open('new_question.csv','a',newline='')
                    content=csv.writer(file,dialect='excel')
                    content.writerow(new_train)
                        
                        
                    ###give final answer###
                else:
                    ###print answer###
                    number = random.randint(1, Answer_set_lengths[int(y_predict)])
                    y_predict = float(y_predict)
                    r_n = number/10
                    y_pre = y_predict + r_n
                    #print(y_pre)
                    Final_answer = Answer_set[y_pre]
                    print('Answer:' + ''.join(Final_answer))
                        
                        ###speak answer###
                    speak.Speak(Final_answer)
                    if y_predict==31.0:
                        isBreak=True
            if isBreak==True:
                break