# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 20:22:26 2018

@author: LilJun
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn import tree
from sklearn.metrics import accuracy_score

csv_file = "ks-projects-201801_1.csv"  # 讀入 csv 檔案

file_encoding = 'utf8'        # set file_encoding to the file encoding (utf8, latin1, etc.)
input_fd = open(csv_file, encoding=file_encoding, errors = 'backslashreplace')
ksr=pd.read_csv(input_fd)
data=pd.DataFrame(ksr)

X = data.drop('state', axis=1)  #X是特徵值
y = data.state   #Y是目標

del data['name']  #刪除不必要測試欄位
#print(data.info()) 


d = defaultdict(LabelEncoder)
X_trans = X.apply(lambda x: d[x.name].fit_transform(x).astype(str))
X_trans.head()

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_trans, y, random_state=1)


print("Train、Test data:")

print ("Train_x Shape : ", X_train.shape)
print ("Train_y Shape : ", y_train.shape)
print ("Test_x Shape : ", X_test.shape)
print ("Test_y Shape : ", y_test.shape)


###隨機森林

print("Decision Tree:")

def decision_tree(features, target):
    clf = tree.DecisionTreeClassifier(max_depth = 3)
    clf.fit(X_train, y_train)
    return clf


decision_model = decision_tree(X_train, y_train)  #call function 
print ("Trained model : ", decision_model)
decision_predictions = decision_model.predict(X_test)
print ("Train Accuracy : ", accuracy_score(y_train,decision_model.predict(X_train)))
print ("Test Accuracy  : ", accuracy_score(y_test, decision_predictions))

#畫出Decision Tree
with open("lc-is.dot", 'w') as f:
     f = tree.export_graphviz(decision_model,
                              out_file=f,
                              impurity = True,
                              feature_names = list(X_train),
                              class_names = ['successful', 'failed','canceled','undefined','live','suspended'],
                              rounded = True,
                              filled= True )

from subprocess import check_call
check_call(['dot','-Tpng','lc-is.dot','-o','lc-is.png'])

from IPython.display import Image as PImage
from PIL import Image, ImageDraw
img = Image.open("lc-is.png")

#draw = ImageDraw.Draw(img)
#img.save('output.png')
#PImage("output.png")


###隨機森林

print("Random Forest:")

from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(criterion='entropy',n_estimators=10,random_state=1,n_jobs=2)



def random_forest(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

random_model = random_forest(X_train, y_train) #call function
print ("Trained model : ", random_model)

predictions = random_model.predict(X_test)

print ("Train Accuracy : ", accuracy_score(y_train,random_model.predict(X_train)))
print ("Test Accuracy  : ", accuracy_score(y_test, predictions))
