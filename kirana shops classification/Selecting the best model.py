# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:04:18 2018

@author: Administrator
"""
##prdiction of y values of test
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

y_pred= model.predict(X_test)
b = np.zeros_like(y_pred)
b[np.arange(len(y_pred)), y_pred.argmax(1)] = 1

##making confusion matrix
indice_of_y_test = y_test.argmax(axis=1)
indice_of_y_test_pred = b.argmax(axis=1)
confusionmatrix = confusion_matrix(indice_of_y_test, indice_of_y_test_pred)

# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range("num of Epochs")

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

