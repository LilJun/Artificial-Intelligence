# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:02:50 2018

@author: LilJun
"""

#資料預處理
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
np.random.seed(10)

 #讀取MNIST資料
(x_train, y_train),(x_test,y_test) = mnist.load_data()

#將features（數字影像特徵值）以reshape轉換爲6000*28*28*1的4維矩陣
x_train4D = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test4D = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')

#將features標準化，可以提高模型預測的準確度，並且更快收斂
x_train4D_normalize = x_train4D / 255
x_test4D_normalize = x_test4D / 255

#使用np_utils.to_categorical, 將訓練資料集與測試的label,進行 Onehot encoding 轉換
y_trainOneHot = np_utils.to_categorical(y_train)
y_testOneHot = np_utils.to_categorical(y_test)

#建立模型
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
model = Sequential()

#建立卷積層1.
#輸入的數字影像是28*28大小，執行第一次卷積運算，會產生16個影像，卷積運算並不會改變影像大小，所以仍然是28*28大小。
model.add(Conv2D(filters=16,
                kernel_size=(5,5),
                padding='same',#補零
                input_shape=(28,28,1),
                activation='relu'))

#建立池化層
#輸入參數pool_size=(2,2),執行第一次縮減取樣，將16個28*28影像，縮小爲16個14*14的影像。
model.add(MaxPooling2D(pool_size=(2,2)))

#建立卷積層2.
#輸入的數字影像是28*28大小，執行第2次卷積運算，將原本16個的影像，轉換爲36個影像，卷積運算並不會改變影像大小，所以仍然是14*14大小。
model.add(Conv2D(filters=36,
                kernel_size=(5,5),
                padding='same',#補零
                activation='relu'))

#建立池化層2
#輸入參數pool_size=(2,2),執行第2次縮減取樣，將36個14*14影像，縮小爲36個7*7的影像。
model.add(MaxPooling2D(pool_size=(2,2)))

#加入Dropout(0.25)層至模型中。其功能是，每次訓練迭代時，會隨機的在神經網絡中放棄25%的神經元，以避免overfitting。
model.add(Dropout(0.25))

#建立平坦層
#之前的步驟已經建立池化層2，共有36個7*7影像，轉換爲1維的向量，長度是36*7*7=1764，也就是1764個float數字，正好對應到1764個神經元。
model.add(Flatten())

#建立隱藏層，共有128個神經元
model.add(Dense(128,activation='relu'))

#加入Dropout(0.5)層至模型中。其功能是，每次訓練迭代時，會隨機的在神經網絡中放棄50%的神經元，以避免overfitting。
model.add(Dropout(0.5))

#建立輸出層
#共有10個神經元，對應到0-9共10個數字。並且使用softmax激活函數進行轉換，softmax可以將神經元的輸出，轉換爲預測每一個數字的機率。
model.add(Dense(10,activation='softmax'))

#查看模型的摘要
print(model.summary())

#定義訓練方式
model.compile(loss='categorical_crossentropy',
             optimizer='adam',metrics=['accuracy'])

#開始訓練
train_history=model.fit(x=x_train4D_normalize,
                       y=y_trainOneHot,validation_split=0.2,
                       epochs=10,batch_size=300,verbose=2)

#畫出accuracy執行結果
#之前訓練步驟，會將每一個訓練週期的accuracy與loss，記錄在train_history。
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

#用test評估模型準確度
scores = model.evaluate(x_test4D_normalize,y_testOneHot)
print('accuracy:',scores[1])

#進行預測 
prediction=model.predict_classes(x_test4D_normalize)

#預測結果
print(prediction[:10])

#顯示前10筆預測結果
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25:
        num =25
    for i in range(0,num):
        ax = plt.subplot(5,5,1+i)
        ax.imshow(images[idx], cmap='binary')
        title = "label="+str(labels[idx])
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.show()
plot_images_labels_prediction(x_test,y_test,prediction,idx=0)

#顯示混淆矩陣
import pandas as pd
pd.crosstab(y_test,prediction,
           rownames=['label'],colnames=['predict'])