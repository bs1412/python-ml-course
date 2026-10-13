# 牛奶质量预测-基础版
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 数据加载
data = pd.read_csv('milknew.csv')

# 数据预处理
data = data.replace(0, np.nan).dropna()

# 检查特征列是否存在（根据实际数据调整）
print("数据列名：", data.columns.tolist())
features = ['pH', 'Temprature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour']
X = data[features]
y = data['Grade']

# 编码标签
le = LabelEncoder()
y = le.fit_transform(y)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 预测与评估
y_pred = knn.predict(X_test)

# 动态获取类别名称
target_names = le.classes_.tolist()
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=target_names))

# 绘制K值选择曲线（保持你原始代码风格）
k_values = range(1, 20)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    accuracies.append(acc)

plt.figure(figsize=(10,6))
plt.plot(k_values, accuracies, marker='o')
plt.title('最优K值选择')
plt.xlabel('K值')
plt.ylabel('测试准确率')
plt.grid(True)
plt.show()