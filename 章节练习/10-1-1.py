#本章需导入的模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.cm as cm
import warnings
warnings.filterwarnings(action = 'ignore')
%matplotlib inline
plt.rcParams['font.sans-serif']=['SimHei']  #解决中文显示乱码问题
plt.rcParams['axes.unicode_minus']=False
from sklearn import svm
import sklearn.linear_model as LM
import scipy.stats as st
from scipy.optimize import root,fsolve
from sklearn.feature_selection import  VarianceThreshold,SelectKBest,f_classif,chi2
from sklearn.feature_selection import RFE,RFECV,SelectFromModel
from sklearn.linear_model import Lasso,LassoCV,lasso_path,Ridge,RidgeCV
from sklearn.linear_model import enet_path,ElasticNetCV,ElasticNet

plt.figure(figsize=(6,4))
for df1,df2,lt in [(10,10,'--'),(15,20,'-.'),(29,18,':'),(50,50,'-')]:
    x = np.linspace(st.f.ppf(0.001, df1, df2),st.f.ppf(0.99, df1, df2), 100)
    plt.plot(x, st.f.pdf(x, df1, df2),label='df1=%d,df2=%d'%(df1,df2),linestyle=lt)
plt.legend()
plt.title("不同自由度下的F分布",fontdict={'fontsize':12})
plt.xlabel("X",fontdict={'fontsize':12})
plt.ylabel("概率密度",fontdict={'fontsize':12})
plt.ylim((0,1.5))

plt.figure(figsize=(6,4))
for df,lt in [(2,'--'),(4,'-.'),(6,':'),(20,'-')]:
    x = np.linspace(st.chi.ppf(0.001, df),st.chi.ppf(0.999, df), 100)
    plt.plot(x, st.chi.pdf(x, df),label='df=%d'%df,linestyle=lt)
plt.legend()
plt.title("不同自由度下的卡方分布",fontdict={'fontsize':12})
plt.xlabel("X",fontdict={'fontsize':12})
plt.ylabel("概率密度",fontdict={'fontsize':12})