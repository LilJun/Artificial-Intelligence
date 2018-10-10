# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:22:47 2018

@author: LilJun
"""
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

csv_file = "ks-projects-201801_1.csv"  # 讀入 csv 檔案

file_encoding = 'utf8'        # set file_encoding to the file encoding (utf8, latin1, etc.)
input_fd = open(csv_file, encoding=file_encoding, errors = 'backslashreplace')
df=pd.read_csv(input_fd)

de=df.dropna()  #將NaN空值欄位進行刪除動作

df_x = pd.DataFrame([de["usd pledged"],de["usd_pledged_real"],de["backers"]]).T  #df_x為變數值
#df_x = pd.DataFrame(de, columns = ['usd pledged', 'usd_pledged_real', 'backers']) 
df_y = pd.DataFrame(de, columns = ['state']) #df_y為預測值

"""
le = preprocessing.LabelEncoder()
le.fit(df_y['state'])
list(le.classes_)
de["state"]=le.transform(df_y['state']) 
y = de["state"]
"""
#將state字串欄位轉為分類值
label_encoder = preprocessing.LabelEncoder()
de["state"] = label_encoder.fit_transform(df_y["state"])
y = de["state"]


#AdaBoost方法
bdt = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=20,min_samples_leaf=5),algorithm='SAMME.R')

param_test1 = {'n_estimators': range(50,300,50),"learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
gsearch1 = GridSearchCV(bdt,param_test1,cv=10)
gsearch1.fit(df_x,y)#3個變數值df_x去和預測值y做fit

print(gsearch1.best_params_, gsearch1.best_score_)

means = gsearch1.cv_results_['mean_test_score']
stds = gsearch1.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, gsearch1.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
    print()
print(gsearch1.best_params_, gsearch1.best_score_)
