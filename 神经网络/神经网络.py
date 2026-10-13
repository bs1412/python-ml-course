import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.cm as cm
import warnings
warnings.filterwarnings(action = 'ignore')
%matplotlib inline
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.neural_network as net

data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

print("数据维度：", data.shape)
print("\n前5行数据：")
print(data.head())

numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
target = 'NObeyesdad'

data[target] = pd.factorize(data[target])[0]

X = data[numeric_cols]
y = data[target]

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
plt.hist(X['Age'], bins=20, edgecolor='k')
plt.title('年龄分布')
plt.grid(True, linestyle='-.')

plt.subplot(2,2,2)
plt.scatter(X['Weight'], X['Height'], c=y, cmap=cm.jet)
plt.colorbar(label='肥胖等级')
plt.title('体重-身高分布')
plt.grid(True, linestyle='-.')

plt.subplot(2,2,3)
plt.boxplot([X['FCVC'], X['NCP'], X['CH2O']], 
           labels=['蔬菜摄入','主餐次数','饮水量'])
plt.title('饮食特征分布')
plt.grid(True, linestyle='-.')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=123)

plt.figure(figsize=(15,5))

nodes = [5, 10, 20]
for i, n in enumerate(nodes):
    mlp = net.MLPClassifier(hidden_layer_sizes=(n,),
                          activation='relu',
                          random_state=123)
    mlp.fit(X_train, y_train)

    train_acc = mlp.score(X_train, y_train)
    test_acc = mlp.score(X_test, y_test)
    
    plt.subplot(1,3,i+1)
    plt.plot(mlp.loss_curve_)
    plt.title(f'隐藏节点数={n}\n训练准确率:{train_acc:.2f}\n测试准确率:{test_acc:.2f}')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.grid(True, linestyle='-.')

plt.tight_layout()
plt.show()

best_mlp = net.MLPClassifier(hidden_layer_sizes=(20,),
                            activation='relu',
                            random_state=123)
best_mlp.fit(X_train, y_train)

plt.figure(figsize=(8,5))
plt.scatter(X_test['Weight'], X_test['Height'],
           c=best_mlp.predict(X_test), 
           cmap=cm.jet,
           s=50, edgecolor='k')
plt.title('最终模型预测分布\n测试准确率: %.2f'%best_mlp.score(X_test, y_test))
plt.xlabel('体重')
plt.ylabel('身高')
plt.colorbar(label='预测类别')
plt.grid(True, linestyle='-.')
plt.show()