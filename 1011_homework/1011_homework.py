# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 19:11:08 2018

@author: LilJun
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
from sklearn import tree
from statsmodels.stats.contingency_tables import mcnemar
from sklearn import metrics
from sklearn import neighbors
import matplotlib.pyplot as plt


csv_file = "ks-projects-201801_1.csv"  # 讀入 csv 檔案

file_encoding = 'utf8'        # set file_encoding to the file encoding (utf8, latin1, etc.)
input_fd = open(csv_file, encoding=file_encoding, errors = 'backslashreplace')
df=pd.read_csv(input_fd)

de=df.dropna()  #將NaN空值欄位進行刪除動作

df_x = pd.DataFrame([de["usd pledged"],de["usd_pledged_real"],de["backers"]]).T  #df_x為變數值
#df_x = pd.DataFrame(de, columns = ['usd pledged', 'usd_pledged_real', 'backers']) 


#將state字串欄位轉為分類值
label_encoder = preprocessing.LabelEncoder()
encoded_state= label_encoder.fit_transform(de["state"]) #state為預測值

# 切分訓練與測試資料
train_X, test_X, train_y, test_y = train_test_split(df_x, encoded_state, test_size = 0.25)

"""
def cart_tree(features, target):
    clf = tree.DecisionTreeClassifier(criterion='gini')
    clf.fit(features, target)
    return clf
"""
#lR model
def logistic_reg(features, target):
    clf=linear_model.LogisticRegression()
    clf.fit(features,target)
    return clf

#CART model
def cart_tree(features, target):
    clf = tree.DecisionTreeClassifier(criterion='gini')
    clf.fit(features, target)
    return clf
#CART model 預設 k = 5
def knn(features, target):
    clf = neighbors.KNeighborsClassifier()
    clf.fit(features, target)
    return clf

#call lR model
logistic_model = logistic_reg(df_x, encoded_state) 
print("係數:",logistic_model.coef_)
print("截距:",logistic_model.intercept_)
lr_predictions = logistic_model.predict(df_x)
lr_accuracy = logistic_model.score(df_x, encoded_state)
print("LR Test Accuracy  : ",lr_accuracy)

#call CART model
cart_model = cart_tree(train_X, train_y)  #call function 
cart_predictions = cart_model.predict(test_X)
print ("CART Test Accuracy  : ", metrics.accuracy_score(test_y, cart_predictions))
 
#call KNN model
knn_model = knn(train_X, train_y) 
knn_predictions = knn_model.predict(test_X)
print ("KNN Test Accuracy (預設k=5)  : ", metrics.accuracy_score(test_y, knn_predictions))

#選擇適合K值
range = np.arange(1, round(0.2 * train_X.shape[0]) + 1)
accuracies = []

for i in range:
    clf = neighbors.KNeighborsClassifier(n_neighbors = i)
    knn_clf = clf.fit(train_X, train_y)
    test_y_predicted = knn_clf.predict(test_X)
    accuracy = metrics.accuracy_score(test_y, test_y_predicted)
    accuracies.append(accuracy)

#選擇適合K值 視覺化
plt.scatter(range, accuracies)
plt.show()
#ans = accuracies.index(max(accuracies)) + 1

ans_k=max(accuracies)
ans=accuracies.index(ans_k)+1
print("k=",ans)
print("Best KNN Test Accuracy:",ans_k)

 