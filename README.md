## This repository contains all the Machine learning projects I have done.
## **Contents**
- [*Kirana* shops classification](#kirana-shops-classification)
- Jigsaw toxic comments classification
- NLP with disaster tweets
- Bitcoin time series prediction
- Credit card clustering

## **Kirana shops classification**
The project's task is to classify *kirana* shops images in 5 different business categories i.e. - 
1. Apparels
2. Pharmaceuticals
3. Electronics
4. Grocery
5. Food items

For classifying images, I implemented Convolutional Neural Network(CNN) model built on Keras and Tensorflow. I performed **[Transfer Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)** method on **[Places365](http://places2.csail.mit.edu/)** dataset. 
I used [VGG-16](https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c) arctitecture which was trained on Places365 dataset and fine-tuned it on ~8000 *kirana* shop images. The best model had the accuracy of ~80%.





