# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 12:47:23 2018

@author: Administrator
"""
from keras import backend as K
import numpy as np
import os
from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

## different for every model
def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.common.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 104.006
        x[:, 1, :, :] -= 116.669
        x[:, 2, :, :] -= 122.679
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 104.006
        x[:, :, :, 1] -= 116.669
        x[:, :, :, 2] -= 122.679
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x

##getting the shape of images
img_path = 'E:/Shop_images/apparels/1.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
print (x.shape)
x = np.expand_dims(x, axis=0)
print (x.shape)
x = preprocess_input(x)
print('Input image shape:', x.shape)
## there are subfolders in test
data_path = 'E:/Shop_images'
data_dir_list = os.listdir(data_path)

img_data_list=[]
img_data_name_list = []

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+dataset)
    for img in img_list:
        img_data_name_list.append(img)
        img_path=data_path+'/'+dataset+'/'+img
        img=image.load_img(img_path,target_size=(224,224))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        img_data_list.append(x)

img_data = np.array(img_data_list)
#pickle.dump(img_data_name_list,open("img_data_name_list.p","wb"))
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)
print (img_data)

num_of_samples=10,060
num_classes = 5
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
labels[0:2010]=0
labels[2010:4134]=1
labels[4134:6543]=2
labels[6543:8010]=3
labels[8010:10060]=4

names = ['apparels','digi','electronics','grocery','pharma']
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#import pickle
#pickle.dump(Y,open("Y.p","wb"))

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
X_train_final, X_dev, y_train_final, y_dev = train_test_split(X_train, y_train, test_size=0.1666, random_state=2)

#proportion_of_train= y_train.sum(axis=0)
#proportion_of_dev= y_dev.sum(axis=0)
#proportion_of_test= y_test.sum(axis=0)
#pickle.dump(y_dev,open("y_dev.p","wb"), protocol=4) 
