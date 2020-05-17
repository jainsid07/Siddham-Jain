# -*- coding: utf-8 -*-
"""
Created on Sat May 16 16:27:36 2020

@author: Siddham Jain
"""

import numpy as np
import pandas as pd
import os
import sys
import random
import keras
import tensorflow as tf
import json
from bert import bert_tokenization
sys.path.insert(0, 'E:/kaggle/toxic_comments/uncased_L-12_H-768_A-12')

BERT_PRETRAINED_DIR = 'E:/kaggle/toxic_comments/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12'
print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))


from keras_bert.bert import get_model
from keras_bert.loader import load_trained_model_from_checkpoint
from keras.optimizers import Adam
adam = Adam(lr=2e-5,decay=0.01)
maxlen = 50
print('begin_build')

config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
model = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True,seq_len=maxlen)
model.summary(line_length=120)

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda
from keras.models import Model
import keras.backend as K
import re
import codecs

sequence_output  = model.layers[-6].output
pool_output = Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(sequence_output)
model3  = Model(inputs=model.input, outputs=pool_output)
model3.compile(loss='binary_crossentropy', optimizer=adam)
model3.summary()


def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for i in range(example.shape[0]):
      tokens_a = tokenizer.tokenize(example[i])
      if len(tokens_a)>max_seq_length:
        tokens_a = tokens_a[:max_seq_length]
        longer += 1
      one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
      all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)

nb_epochs=1
bsz = 32
dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
tokenizer = bert_tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=True)
print('build tokenizer done')



train_df = pd.read_csv('E:/kaggle/toxic_comments/train.csv')
train_df = train_df.sample(frac=0.01,random_state = 42)
#train_df['comment_text'] = train_df['comment_text'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n',  ' ', regex=True)

train_lines, train_labels = train_df['comment_text'].values, train_df.target.values 
print('sample used',train_lines.shape)

token_input = convert_lines(train_lines,maxlen,tokenizer)
seg_input = np.zeros((token_input.shape[0],maxlen))
mask_input = np.ones((token_input.shape[0],maxlen))
print(token_input.shape)
print(seg_input.shape)
print(mask_input.shape)
print('begin training')
model3.fit([token_input, seg_input, mask_input],train_labels,batch_size=bsz,epochs=nb_epochs)

# you can save the fine-tuning model by this line.
model3.save_weights('bert_weights.h5')

test_df = pd.read_csv('E:/kaggle/toxic_comments/test.csv')
test_df = test_df.sample(frac=0.01,random_state = 42)

eval_lines = test_df['comment_text'].values
print(eval_lines.shape)
print('load data done')

token_input2 = convert_lines(eval_lines, maxlen, tokenizer)

seg_input2 = np.zeros((token_input2.shape[0], maxlen))
mask_input2 = np.ones((token_input2.shape[0], maxlen))
print('test data done')
print(token_input2.shape)
print(seg_input2.shape)
print(mask_input2.shape)

hehe = model3.predict([token_input2, seg_input2, mask_input2], verbose=1, batch_size=bsz)
