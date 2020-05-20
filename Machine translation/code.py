# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:56:32 2020

@author: Siddham Jain
"""

import string
import re
import numpy as np
from numpy import array, argmax, random, take
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
pd.set_option('display.max_colwidth', 200)

# function to read raw text file
def read_text(filename):
        # open the file
        file = open(filename, mode='rt', encoding='utf-8')
        
        # read all text
        text = file.read()
        file.close()
        return text
    
# split a text into sentences
def to_lines(text):
      sents = text.strip().split('\n')
      sents = [i.split('\t') for i in sents]
      return sents

#reading data
data = read_text("E:/kaggle/machine-translation/deu.txt")
ger_to_eng = to_lines(data)
ger_to_eng = array(ger_to_eng)

# Remove punctuation
ger_to_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in ger_to_eng[:,0]]
ger_to_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in ger_to_eng[:,1]]

# convert text to lowercase
for i in range(len(ger_to_eng)):
    ger_to_eng[i,0] = ger_to_eng[i,0].lower()
    ger_to_eng[i,1] = ger_to_eng[i,1].lower()
    
# empty lists
eng_l = []
ger_l = []

# populate the lists with sentence lengths
for i in ger_to_eng[:,0]:
      eng_l.append(len(i.split()))

for i in ger_to_eng[:,1]:
      ger_l.append(len(i.split()))

length_df = pd.DataFrame({'eng':eng_l, 'ger':ger_l})


# function to build a tokenizer
def tokenization(lines):
      tokenizer = Tokenizer()
      tokenizer.fit_on_texts(lines)
      return tokenizer

# prepare english tokenizer
eng_tokenizer = tokenization(ger_to_eng[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1

eng_length = 8
print('English Vocabulary Size: %d' % eng_vocab_size)

# prepare Deutch tokenizer
ger_tokenizer = tokenization(ger_to_eng[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1

ger_length = 8
print('German Vocabulary Size: %d' % ger_vocab_size)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
         # integer encode sequences
         seq = tokenizer.texts_to_sequences(lines)
         # pad sequences with 0 values
         seq = pad_sequences(seq, maxlen=length, padding='post')
         return seq
     
from sklearn.model_selection import train_test_split

# split data into train and test set
train, test = train_test_split(ger_to_eng, test_size=0.2, random_state = 12)

# prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])

# prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])

# build NMT model
def define_model(in_vocab,out_vocab, in_timesteps,out_timesteps,units):
      model = Sequential()
      model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
      model.add(LSTM(units))
      model.add(RepeatVector(out_timesteps))
      model.add(LSTM(units, return_sequences=True))
      model.add(Dense(out_vocab, activation='softmax'))
      return model
  
# model compilation
model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 512)
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# train model
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
                    epochs=2, batch_size=512, validation_split = 0.2, 
                    verbose=1)

preds = model.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))

def get_word(n, tokenizer):
      for word, index in tokenizer.word_index.items():
          if index == n:
              return word
      return None
  
preds_text = []
for i in preds:
       temp = []
       for j in range(len(i)):
            t = get_word(i[j], eng_tokenizer)
            if j > 0:
                if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                     temp.append('')
                else:
                     temp.append(t)
            else:
                   if(t == None):
                          temp.append('')
                   else:
                          temp.append(t) 

       preds_text.append(' '.join(temp))

pred_df = pd.DataFrame({'actual' : test[:,0], 'predicted' : preds_text})

