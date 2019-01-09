# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 09:44:15 2019

@author: DELL
"""

import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

num_classes = 10
epochs = 6

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def cnn(activation = 'relu'):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation=activation,
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    return model.fit(x_train, y_train,
                     batch_size=128,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(x_test, y_test))

relu_history = cnn(activation = 'relu')
sigmoid_history = cnn(activation = 'sigmoid')

print('1. Sigmoid v.s. ReLu')
plt.figure(figsize=(10, 7))
plt.suptitle('relu v.s sigmoid')

plt.subplot(2, 2, 1)
plt.plot(range(1, epochs + 1), relu_history.history['loss'], label = 'relu')
plt.plot(range(1, epochs + 1), sigmoid_history.history['loss'], label = 'sigmoid')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(1, epochs + 1), relu_history.history['acc'], label = 'relu')
plt.plot(range(1, epochs + 1), sigmoid_history.history['acc'], label = 'sigmoid')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(range(1, epochs + 1), relu_history.history['val_loss'], label = 'relu')
plt.plot(range(1, epochs + 1), sigmoid_history.history['val_loss'], label = 'sigmoid')
plt.xlabel('epochs')
plt.ylabel('val_loss')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(range(1, epochs + 1), relu_history.history['val_acc'], label = 'relu')
plt.plot(range(1, epochs + 1), sigmoid_history.history['val_acc'], label = 'sigmoid')
plt.xlabel('epochs')
plt.ylabel('val_acc')
plt.legend()
plt.show()