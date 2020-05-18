# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 12:51:56 2018

@author: Administrator
"""


import time
## pre-trained model on which we want to do transfer learning 
image_input = Input(shape=(224, 224, 3))
model = VGG16_Places365(input_tensor=image_input, weights='places')
model.summary()
## after this layer we want to add layers
last_layer = model.get_layer('drop_fc2').output
#adding(changing) the layers after fc2
x = Dense(1024, activation='relu',kernel_initializer="glorot_uniform", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4), name='fc3')(last_layer)
x = Dropout(0.5, name='drop_fc3')(x)
x = Dense(256, activation='relu',kernel_initializer="glorot_uniform", name='fc4')(x)
x = Dropout(0.5, name='drop_fc4')(x)
out = Dense(5, activation='softmax', name='output')(x)
custom_vgg_model_1 = Model(image_input, out)
custom_vgg_model_1.summary()
#defining no. of parameters to be trained
for layer in custom_vgg_model_1.layers[:-5]:
	layer.trainable = False

custom_vgg_model_1.summary()

#import pickle
#y_test = pickle.load( open( "y_test.p", "rb" ) )

##running the new model
adm = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
custom_vgg_model_1.compile(loss='categorical_crossentropy',optimizer=adm ,metrics=['accuracy'])
t=time.time()

hist = custom_vgg_model_1.fit(X_train, y_train, batch_size=32, epochs=5, verbose=1, validation_data=(X_dev, y_dev))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model_1.evaluate(X_test, y_test, batch_size=10, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

#saving a model
modelvgg2_json = custom_vgg_model_1.to_json()
with open("custom_vgg_model2.json", "w") as json_file:
    json_file.write(modelvgg2_json)
# serialize weights to HDF5
custom_vgg_model_1.save_weights("model2.h5")
print("Saved model to disk")