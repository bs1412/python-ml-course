import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

class KMeans:
    def __init__(self, k=3, max_iters=300, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia = None
        
    def _initialize_centroids(self, X):
        # 随机初始化质心
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices]
    
    def _assign_clusters(self, X, centroids):
        # 将样本分配到最近的质心
        labels, distances = pairwise_distances_argmin_min(X, centroids)
        self.inertia = np.sum(distances ** 2)
        return labels
    
    def _update_centroids(self, X, labels):
        # 更新质心位置为簇内点的均值
        new_centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
        return new_centroids
    
    def fit(self, X):
        # 初始化质心
        self.centroids = self._initialize_centroids(X)
        prev_inertia = float('inf')
        
        # 迭代优化
        for _ in range(self.max_iters):
            # 分配样本到最近的簇
            labels = self._assign_clusters(X, self.centroids)
            
            # 检查是否收敛
            if prev_inertia - self.inertia < self.tol:
                break
            prev_inertia = self.inertia
            
            # 更新质心
            new_centroids = self._update_centroids(X, labels)
            
            # 检查质心是否变化
            if np.allclose(new_centroids, self.centroids, rtol=self.tol):
                break
                
            self.centroids = new_centroids
        
        # 最终分配
        self.labels = self._assign_clusters(X, self.centroids)
        
        return self
    
    def predict(self, X):
        # 预测新样本的簇标签
        labels, _ = pairwise_distances_argmin_min(X, self.centroids)
        return labels