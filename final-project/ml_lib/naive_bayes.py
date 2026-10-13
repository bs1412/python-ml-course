import numpy as np
from collections import defaultdict

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.mean = {}
        self.var = {}
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        
        # 计算每个类别的先验概率
        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = X_c.shape[0] / X.shape[0]
            
            # 计算每个特征的均值和方差
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-9  # 添加小值防止除零错误
        
    def _gaussian_pdf(self, x, mean, var):
        # 计算高斯概率密度函数
        exponent = np.exp(-((x - mean)**2) / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent
    
    def predict(self, X):
        y_pred = []
        
        for sample in X:
            posteriors = []
            
            # 计算每个类别的后验概率
            for c in self.classes:
                prior = np.log(self.class_priors[c])  # 使用对数防止数值下溢
                likelihood = 0
                
                # 假设特征独立，计算联合概率
                for i, feature in enumerate(sample):
                    likelihood += np.log(self._gaussian_pdf(
                        feature, self.mean[c][i], self.var[c][i]
                    ))
                
                posteriors.append(prior + likelihood)
            
            # 选择后验概率最大的类别
            y_pred.append(self.classes[np.argmax(posteriors)])
        
        return np.array(y_pred)