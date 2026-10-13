import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

try:
    data = pd.read_csv('milknew.csv', encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv('milknew.csv', encoding='GBK')

print("原始数据样例：\n", data.head())
print("\n初始数据量：", len(data))

data = data[(data['Temprature'] <= 120) & (data['Temprature'] >= 20)]
print("\n温度过滤后数据量：", len(data))

data = data.replace(0, np.nan).dropna()
print("缺失值处理后数据量：", len(data))

le = LabelEncoder()
data['Grade'] = le.fit_transform(data['Grade'])

class_counts = data['Grade'].value_counts().sort_index()
print("\n类别分布：")
for cls, count in zip(le.classes_, class_counts):
    print(f"{cls}: {count}个样本")

features = ['pH', 'Temprature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour']
X = data[features]
y = data['Grade']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    train_size=0.7,
    random_state=42,
    stratify=y
)

max_k = int(len(X_train) * 0.2)
k_values = range(1, max_k, 2) 
test_scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model.fit(X_train, y_train)
    test_scores.append(model.score(X_test, y_test))

best_k = k_values[np.argmax(test_scores)]

plt.figure(figsize=(10, 6))
plt.plot(k_values, test_scores, 'bo-', markersize=8)
plt.title(f'K值选择曲线 (最优K={best_k})')
plt.xlabel('K值')
plt.ylabel('测试准确率')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('knn_performance.png', dpi=300)
plt.close()

final_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

target_names = le.classes_

print("\n================ 分类报告 ================")
print(classification_report(y_test, y_pred, target_names=target_names))

with open('analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(f"最佳K值: {best_k}\n")
    f.write("分类报告:\n")
    f.write(classification_report(y_test, y_pred, target_names=target_names))

