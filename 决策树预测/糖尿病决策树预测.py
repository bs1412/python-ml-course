import numpy as np
from collections import Counter

# 简易版决策树分类器
class SimpleDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    # 计算基尼系数
    def _gini(self, labels):
        counts = Counter(labels)
        return 1 - sum((v/len(labels))**2 for v in counts.values())

    # 寻找最佳分割点
    def _find_split(self, X, y):
        best_gain = 1
        best_feature, best_val = None, None
        
        for col in range(X.shape[1]):
            values = np.unique(X[:, col])
            for val in values:
                left_mask = X[:, col] <= val
                if sum(left_mask) == 0 or sum(~left_mask) == 0:
                    continue
                
                # 计算加权基尼系数
                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[~left_mask])
                total_gini = (left_gini * sum(left_mask) + 
                             right_gini * sum(~left_mask)) / len(y)
                
                if total_gini < best_gain:
                    best_gain = total_gini
                    best_feature = col
                    best_val = val
        return best_feature, best_val

    # 递归构建树
    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(set(y)) == 1:
            return Counter(y).most_common(1)[0][0]
        
        feature, val = self._find_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]
        
        left_mask = X[:, feature] <= val
        left = self._build_tree(X[left_mask], y[left_mask], depth+1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth+1)
        
        return (feature, val, left, right)

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        node = self.tree
        while isinstance(node, tuple):
            feat, val, left, right = node
            node = left if x[feat] <= val else right
        return node

# 数据预处理（简单填充缺失值）
data = np.genfromtxt('糖尿病.csv', delimiter=',', skip_header=1)
X = data[:, :-1]
y = data[:, -1].astype(int)

# 用均值填充0值
for col in [1,2,3,4,5]:  # 选取需要处理的列
    X[X[:, col]==0, col] = np.mean(X[X[:, col]!=0, col])

# 划分训练测试集
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# 训练模型
model = SimpleDecisionTree(max_depth=3)
model.fit(X_train, y_train)

# 基础版评估函数
def simple_evaluate(y_true, y_pred):
    accuracy = sum(y_true == y_pred) / len(y_true)
    print(f"准确率: {accuracy:.4f}")
    
    # 简易版分类报告
    classes = np.unique(y_true)
    print("\n类\t精确率\t召回率\tF1")
    for c in classes:
        tp = sum((y_pred == c) & (y_true == c))
        fp = sum((y_pred == c) & (y_true != c))
        fn = sum((y_pred != c) & (y_true == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{c}\t{precision:.2f}\t{recall:.2f}\t{f1:.2f}")

# 输出结果
preds = model.predict(X_test)
simple_evaluate(y_test, preds)