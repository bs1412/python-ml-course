import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)
    
    def _grow_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)
        
        # 终止条件
        if (len(unique_classes) == 1 or 
            (self.max_depth is not None and depth >= self.max_depth) or
            num_samples < 2):
            return Node(value=self._most_common_label(y))
        
        # 寻找最佳分割
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(num_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth+1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def _information_gain(self, X, y, feature_idx, threshold):
        parent_entropy = self._entropy(y)
        
        left_idxs = X[:, feature_idx] <= threshold
        right_idxs = ~left_idxs
        
        n = len(y)
        n_left, n_right = np.sum(left_idxs), np.sum(right_idxs)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        child_entropy = (n_left/n) * self._entropy(y[left_idxs]) + \
                        (n_right/n) * self._entropy(y[right_idxs])
        
        # 信息增益
        gain = parent_entropy - child_entropy
        return gain
    
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)