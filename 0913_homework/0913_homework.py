# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures # 用來將X轉成ax+b特徵才能回歸



x=np.linspace(0,2*np.pi,100) #從0到X之間產生100個點
y=np.sin(x)#產生一個sin函數
plt.subplot(2,1,1)#subplot 子圖(2,1,1)2列1行  最後一個值為當前位置
plt.scatter(x,y)#畫出17行散佈圖

x1=np.linspace(0,2*np.pi,100) #樣本數
y1=np.sin(x1)+np.random.randn(len(x1))/5.0
plt.subplot(3,1,1)
plt.scatter(x1,y1)

slr=LinearRegression() #類似一個類別
#x1=pd.DataFrame(x1)
x1=x1.reshape(-1,1) #reshape將一維轉維二維 ，多一個維度
slr.fit(x1,y1) #找到輸入和輸出之間的關係，x1指有一個 (用fit做訓練)
print("回歸係數",slr.coef_)
print("截距",slr.intercept_)
predicted_y1=slr.predict(x1) #利用訓練完的函數帶進去得到預測的y1，就是計算的y1
plt.subplot(3,1,1)
plt.plot(x1,predicted_y1) #產生回歸圖

poly_features_3=PolynomialFeatures(degree=3,include_bias=False)
X_poly_3=poly_features_3.fit_transform(x1)
lin_reg_3=LinearRegression()
lin_reg_3.fit(X_poly_3,y1)
print(lin_reg_3.intercept_,lin_reg_3.coef_)

X_plot=np.linspace(0,6,1000).reshape(-1,1)
X_plot_poly=poly_features_3.fit_transform(X_plot)
y_plot=np.dot(X_plot_poly,lin_reg_3.coef_.T)+lin_reg_3.intercept_
plt.subplot(3,1,2)
plt.plot(X_plot,y_plot,'r-')
plt.plot(x1,y1,'b.')


