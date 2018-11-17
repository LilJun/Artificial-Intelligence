# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:57:20 2018

@author: LilJun
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

csv_file = "ks-projects-201801_1.csv"  # 讀入 csv 檔案

file_encoding = 'utf8'        # set file_encoding to the file encoding (utf8, latin1, etc.)
input_fd = open(csv_file, encoding=file_encoding, errors = 'backslashreplace')
df=pd.read_csv(input_fd)

de=df.dropna()  #將NaN空值欄位進行刪除動作

X = de.drop('state', axis=1)   #除了預測值:state其餘變數皆為X變數值
#df_x = pd.DataFrame([de["usd pledged"],de["usd_pledged_real"],de["backers"]]).T  #df_x為變數值


d = defaultdict(LabelEncoder)
X_trans = X.apply(lambda x: d[x.name].fit_transform(x).astype(str))  #將變數值X進行字串轉換

#將state字串欄位轉為分類值
label_encoder = preprocessing.LabelEncoder()
encoded_state= label_encoder.fit_transform(de["state"]) #state為預測值


#將資料切割為訓練集60%，測試集40%
train_X, test_X, train_y, test_y = train_test_split(X_trans, encoded_state, test_size = 0.4,random_state=0)

#使用naive_bayes
gnb = GaussianNB() 
gnb.fit(train_X, train_y) 
  
# making predictions on the testing set 
y_pred = gnb.predict(test_X) 
  
# comparing actual response values (y_test) with predicted response values (y_pred) 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(test_y, y_pred)*100)

