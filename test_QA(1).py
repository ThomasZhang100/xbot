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

num_answers=31
###read answers dataset###
data = pd.read_csv(r'ChatbotAnswersTXT.txt',sep = ":")
Answers = data['ANSWER'].values
IDs = data['ID'].values

###build answers dictionary###
# print(Answers)
Answer_set = {} #dictionary to list
for i in IDs:
    Answer_set[i]=[]
for i in range(len(Answers)):  
    Answer_set[IDs[i]].append(Answers[i])
# print(Answer_set)

###speak welcome###
speak = wincl.Dispatch("SAPI.SpVoice")
speak.Speak("Hello, how can I help you")

###speech recognition###
#sample_rate = 48000
#chunk_size = 2048
#r.dynamic_energy_threshold = True
#r.dynamic_energy_adjustment_damping = 0.15
#r.dynamic_energy_adjustment_ratio = 1.2
model=load_model('QACNNmodel1(1).h5')
model.summary()
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
        
        ###quit loop###
        if recog == Answers[16]:
            gb=["Goodbye, thank you"]
            print('Answer:' + ''.join(gb))
            speak.Speak(gb)
            break
        
        else:
            print("Read question")
            test_data = []
            test_data.append(recog)
            
            ###load dictionary###
            dic = vocabulary1.dictionary
            #print(dic)
            ###one hot embeding###
            max_length = 24
            x_test = np.zeros(shape=(len(test_data),
                                     max(dic.values()) + 1,
                                     max_length)) 
            print(max(dic.values()) + 1)
            for i, test_data in enumerate(test_data):  
                for j, word in list(enumerate(test_data.split()))[:max_length]:
                    index = dic.get(word)
                    x_test[i,index,j] = 1
            #print(x_test)
            ###load the model###      
            
            
            ###predict class###
            y_predict = model.predict_classes(x_test)
            print(y_predict)
        
            
            ###probability of prediction###
            y_probability = model.predict_proba(x_test)
            #print(y_probability)
            
            ###quit the loop###
            if y_predict == 7:
                y_predict7 = float(y_predict)
                number7 = random.randint(1,2)
                r_n7 = number7/10
                y_pre7 = y_predict7 + r_n7
                Final_answer7 = Answer_set[y_pre7]
                print('Answer:' + ''.join(Final_answer7))
                speak.Speak(Final_answer7)
                break
            
            ###current date### 
            elif y_predict == 8:
                current_date = datetime.datetime.now()
                date_answer = current_date.strftime('%Y/%m/%d')
                print('Answer:' + 'Today is ' + ''.join(date_answer))
                speak.Speak('Today is ' + date_answer)
            
            ###current time###
            elif y_predict == 9:
                current_time = datetime.datetime.now()
                time_answer = current_time.strftime('%H:%M')
                print('Answer:' + 'It is currently ' + ''.join(time_answer))
                speak.Speak('It is currently' + time_answer)
                
            else:
                ###save unknow questions###
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
                        y_predict=int(y_predict)
                        number = random.randint(1,len(Answer_set[y_predict])-1)
                        Final_answer = Answer_set[y_predict][number]
                        print('Answer:' + ''.join(Final_answer))
                        
                        ###speak answer###
                        speak.Speak(Final_answer) 
