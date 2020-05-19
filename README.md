## This repository contains all the Machine learning projects I have done.
## **Contents**
- [*Kirana* shops classification](#kirana-shops-classification)
- [Jigsaw toxic comments classification](#jigsaw-toxic-comments-classification)
- NLP with disaster tweets
- Bitcoin time series prediction
- Credit card clustering

## [**Kirana shops classification**](https://github.com/jainsid07/Siddham-Jain/tree/master/kirana%20shops%20classification)
The project's task is to classify *kirana* shops images in 5 different business categories i.e. - 
1. Apparels
2. Pharmaceuticals
3. Electronics
4. Grocery
5. Food items

For classifying images, Convolutional Neural Network (CNN) model built on Keras and Tensorflow was implemented. I used the method of [Transfer Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/) on [Places365](http://places2.csail.mit.edu/) dataset which has images very similar to the shop images. 
I used pre-trained [VGG-16](https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c) arctitecture and fine-tuned it on ~10,000 *kirana* shop images. The best model had the accuracy of ~80%.

## **Jigsaw toxic comments classification**
The project's task is to classify the comments as toxic or non-toxic. For implementing that, following models were implemented - 
1. Simple RNN
2. LSTM
3. GRU
4. Bi-directional RNN
5. Encoder-decoder models
6. Attention models
7. Transformers
8. BERT

You can find the information about each model [here](https://github.com/jainsid07/Siddham-Jain/blob/master/Jigsaw--toxic-comments-classification/ReadMe)







