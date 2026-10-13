# 手动实现朴素贝叶斯（高斯分布版）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- 1. 加载数据 -----------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None, names=[
    "sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"
])

# ----------------- 2. 数据预处理 -----------------
# 随机打乱数据
df = df.sample(frac=1).reset_index(drop=True)
# 分割训练集和测试集（70%训练，30%测试）
train_size = int(0.7 * len(df))
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# ----------------- 3. 手动实现贝叶斯 -----------------
class NaiveBayes:
    def __init__(self):
        self.class_info = {}  # 存储每个类的统计信息
    
    def fit(self, X, y):
        # 获取所有类别
        self.classes = np.unique(y)
        for c in self.classes:
            # 筛选当前类别的数据
            X_c = X[y == c]
            # 计算每个特征的均值和方差
            mean = X_c.mean(axis=0)
            var = X_c.var(axis=0) + 1e-9  # 防止零方差
            # 存储先验概率和分布参数
            self.class_info[c] = {
                'prior': len(X_c)/len(X),
                'mean': mean,
                'var': var
            }
    
    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            posteriors = []
            # 对每个类别计算后验概率
            for c in self.classes:
                prior = np.log(self.class_info[c]['prior'])  # 使用对数防止数值下溢
                # 计算高斯概率
                likelihood = np.sum(
                    -0.5 * np.log(2 * np.pi * self.class_info[c]['var']) 
                    - 0.5 * ((row - self.class_info[c]['mean'])**2 / self.class_info[c]['var'])
                )
                posteriors.append(prior + likelihood)
            # 选择概率最大的类别
            predictions.append(self.classes[np.argmax(posteriors)])
        return predictions

# ----------------- 4. 训练和预测 -----------------
# 准备数据
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_true = test_data.iloc[:, -1]

# 训练模型
model = NaiveBayes()
model.fit(X_train, y_train)

# 预测结果
y_pred = model.predict(X_test)

# ----------------- 5. 评估结果 -----------------
accuracy = np.sum(y_true.values == y_pred) / len(y_test)
print(f"模型准确率: {accuracy:.2%}")

# ----------------- 6. 可视化 -----------------
# 特征分布图
plt.figure(figsize=(10,6))
for idx, feature in enumerate(df.columns[:-1]):
    plt.subplot(2,2,idx+1)
    for c in model.classes:
        plt.hist(df[df['class']==c][feature], alpha=0.5, label=c)
    plt.title(feature)
    plt.legend()
plt.tight_layout()
plt.savefig('feature_dist.png')

# 混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(model.classes))
plt.xticks(tick_marks, model.classes, rotation=45)
plt.yticks(tick_marks, model.classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')
