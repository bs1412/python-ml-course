import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        # 计算距离
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # 获取最近的k个样本
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        # 投票
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]