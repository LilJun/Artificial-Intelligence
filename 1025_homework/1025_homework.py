# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:03:20 2018

@author: LilJun
"""


import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing


#載入資料集
csv_file = "ks-projects-201801_1.csv"  # 讀入 csv 檔案

file_encoding = 'utf8'        # set file_encoding to the file encoding (utf8, latin1, etc.)
input_fd = open(csv_file, encoding=file_encoding, errors = 'backslashreplace')
preksr=pd.read_csv(input_fd)

ksr=preksr.dropna()  #將NaN空值欄位進行刪除動作
df_x = pd.DataFrame(ksr,columns=["usd pledged","usd_pledged_real", "backers"])  #df_x為變數值

#將state字串欄位轉為分類值
label_encoder = preprocessing.LabelEncoder()
encoded_state= label_encoder.fit_transform(ksr["state"]) #state為預測值

# 切分訓練與測試資料

train_X, test_X, train_y, test_y = train_test_split(df_x, encoded_state, test_size = 0.3,random_state=0)


def svc_validation(train_x, train_y,test_x,test_y):
  
    model = SVC(kernel='rbf', probability=True) 
    grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, grid, n_jobs = 1, verbose=1)
    grid_search.fit(train_x, train_y)
    for params, mean_score, scores in grid_search.grid_scores_:
       model = SVC(kernel='rbf', C=params['C'], gamma=params['gamma'], probability=True)
       model.fit(train_x, train_y)
       predict = model.predict(test_x)
       accuracy = metrics.accuracy_score(test_y, predict)
       print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params),"accuracy: %.2f%%" % (100 * accuracy))
        
        
    best_parameters = grid_search.best_estimator_.get_params()
    print("\nbest parameters:\n",best_parameters)
    for para, val in best_parameters.items():
       # print (para, val)
       model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
       model.fit(train_x, train_y)
    return model


model = svc_validation(train_X,train_y,test_X,test_y)  #call function
predict = model.predict(test_X)
accuracy = metrics.accuracy_score(test_y, predict)
print ('\ntest best accuracy: %.2f%%' % (100 * accuracy))

predict=model.predict(train_X)
accuracy = metrics.accuracy_score(train_y, predict)
print ('\ntrain best accuracy: %.2f%%' % (100 * accuracy))

#print ('\n測試集',model.score(test_X, test_y))
#print ('\n訓練集',model.score(train_X, train_y)) 
