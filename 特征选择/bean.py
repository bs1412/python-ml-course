import numpy as np
import pandas as pd

def anova_f_value(X, y):
    classes = np.unique(y)
    n_features = X.shape[1]
    f_values = np.zeros(n_features)
    
    for feature_idx in range(n_features):
        overall_mean = np.mean(X[:, feature_idx])
        n_total = len(y)
        
        ss_between = 0
        ss_within = 0
        
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            n_cls = len(cls_indices)
            
            if n_cls == 0:
                continue
                
            cls_mean = np.mean(X[cls_indices, feature_idx])
            cls_values = X[cls_indices, feature_idx]
            
            # 组间平方和
            ss_between += n_cls * (cls_mean - overall_mean) ** 2
            
            # 组内平方和
            ss_within += np.sum((cls_values - cls_mean) ** 2)
        
        # 计算自由度
        df_between = len(classes) - 1
        df_within = n_total - len(classes)
        
        # 避免除以零
        if df_within == 0 or ss_within == 0:
            f_values[feature_idx] = 0
        else:
            ms_between = ss_between / df_between
            ms_within = ss_within / df_within
            f_values[feature_idx] = ms_between / ms_within
    
    return f_values

def calc_entropy(x):
    value_counts = np.bincount(x.astype(int))
    probabilities = value_counts / len(x)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def mutual_info(X, y, bins=20):
    n_features = X.shape[1]
    mi_scores = np.zeros(n_features)
    
    # 离散化目标变量
    unique_classes, y_encoded = np.unique(y, return_inverse=True)
    
    for i in range(n_features):
        # 将连续特征离散化
        x_discrete = np.digitize(X[:, i], bins=np.histogram(X[:, i], bins)[1])
        
        # 计算联合分布
        joint_dist = np.zeros((len(np.unique(x_discrete)), len(unique_classes)))
        for val_x in np.unique(x_discrete):
            for val_y in np.unique(y_encoded):
                joint_dist[val_x, val_y] = np.sum((x_discrete == val_x) & (y_encoded == val_y))
                
        joint_dist /= len(y)
        
        # 计算互信息
        hx = calc_entropy(x_discrete)
        hy = calc_entropy(y_encoded)
        hxy = -np.sum([p * np.log2(p) for p in joint_dist.flatten() if p > 0])
        
        # 互信息 = H(x) + H(y) - H(x,y)
        mi_scores[i] = hx + hy - hxy
    
    return mi_scores

def feature_ranking(df, method='anova'):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    if method == 'anova':
        scores = anova_f_value(X, y)
    elif method == 'mutual_info':
        scores = mutual_info(X, y)
    else:
        raise ValueError("Invalid method. Choose 'anova' or 'mutual_info'")
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'Feature': df.columns[:-1],
        'Score': scores
    })
    
    # 按分值降序排序
    return results.sort_values(by='Score', ascending=False)

# 主程序
if __name__ == "__main__":
    # 读取Excel文件
    df = pd.read_csv('Dry_Bean.csv', sep=',')
    
    # 移除第一列序号
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # 方法选择: 'anova' 或 'mutual_info'
    ranking_df = feature_ranking(df, method='anova')
    
    # 输出结果
    print("特征重要度排序(ANOVA F值):")
    print(ranking_df)
    
    # 输出建议选择的特征
    top_features = ranking_df.head(7)['Feature'].tolist()
    print("\n推荐选择的特征(前7个):")
    print(top_features)