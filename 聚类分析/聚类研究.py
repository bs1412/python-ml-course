import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def load_data(file_name):
    data = pd.read_csv(file_name)
    print("共找到", len(data), "条记录")
    return data

def standardize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

class SimpleKMeans:
    def __init__(self, k=3, max_iter=100):
        self.k = k  
        self.max_iter = max_iter  
    
    def fit(self, X):
        centroids = X[random.sample(range(len(X)), self.k)]
        
        for _ in range(self.max_iter):
            distances = np.zeros((len(X), self.k))
            for i in range(self.k):
                distances[:, i] = np.sqrt(((X - centroids[i])**2).sum(axis=1))
            
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros_like(centroids)
            for i in range(self.k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    new_centroids[i] = X[random.randint(0, len(X)-1)]
            
            if np.allclose(centroids, new_centroids):
                print(f"聚类收敛于 {_} 次迭代")
                centroids = new_centroids
                break
                
            centroids = new_centroids
        
        self.centroids = centroids
        self.labels = labels
        return self
    
    def predict(self, X):
        distances = np.zeros((len(X), self.k))
        for i in range(self.k):
            distances[:, i] = np.sqrt(((X - self.centroids[i])**2).sum(axis=1))
        return np.argmin(distances, axis=1)

def analyze_clusters(data, labels):
    data = data.copy()
    data['Cluster'] = labels
    
    cluster_stats = data.groupby('Cluster').agg(
        Size=('Purchased', 'count'),
        PurchaseRate=('Purchased', 'mean')
    )
    
    print("\n聚类分析结果:")
    print("================")
    for cluster_id, row in cluster_stats.iterrows():
        print(f"簇 {cluster_id}:")
        print(f"  样本数量: {row['Size']}")
        print(f"  购买率: {row['PurchaseRate']:.2%}")
        print("-" * 20)
    
    return data

def plot_clusters(X, centroids, labels, show=False):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
    
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='X', label='质心')
    
    plt.title('简化的K-Means聚类结果')
    plt.xlabel('年龄')
    plt.ylabel('薪资')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if show:
        plt.show()
    
    return plt.gcf()

def main():
    data = load_data('Social_Network_Ads.csv')
    
    features = data[['Age', 'EstimatedSalary']].values
    
    features_scaled, mean, std = standardize_features(features)
    
    k = 3  
    
    kmeans = SimpleKMeans(k=k)
    kmeans.fit(features_scaled)
    
    labels = kmeans.labels
    
    data_with_clusters = analyze_clusters(data, labels)
    
    centroids_original = kmeans.centroids * std + mean
    
    fig = plot_clusters(features, centroids_original, labels, show=True)
    
    save_choice = input("是否保存结果图片？(y/n): ")
    if save_choice.lower() == 'y':
        fig.savefig('kmeans_results.png')
        print("结果已保存为 kmeans_results.png")
    
    print("\n分析结论:")
    print("========================")
    print("通过聚类分析，我们将顾客分为了3个主要群体：")
    print("- 簇0: 购买力较低的群体")
    print("- 簇1: 中等购买力的群体")
    print("- 簇2: 购买力较强的群体")
    print("\n这些分组可以帮助我们针对不同群体制定营销策略。")
    print("\n注意：具体结论需要结合业务背景进一步分析。")

if __name__ == "__main__":
    main()