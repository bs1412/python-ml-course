import pandas as pd
import numpy as np

def load_data(file_path):
    """
    加载数据集，支持csv, xlsx格式
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

def handle_missing_values(data, strategy='mean'):
    """
    处理缺失值
    可选策略：'mean'(均值填充), 'median'(中位数填充), 'mode'(众数填充), 'drop'(删除含缺失值行)
    """
    if strategy == 'mean':
        return data.fillna(data.mean())
    elif strategy == 'median':
        return data.fillna(data.median())
    elif strategy == 'mode':
        return data.fillna(data.mode().iloc[0])
    elif strategy == 'drop':
        return data.dropna()
    else:
        raise ValueError("Invalid strategy for missing value handling")

def remove_duplicates(data):
    """删除重复行"""
    return data.drop_duplicates()

def normalize_data(data, method='minmax'):
    """
    数据归一化
    可选方法：'minmax'(0-1归一化), 'zscore'(标准化)
    """
    if method == 'minmax':
        return (data - data.min()) / (data.max() - data.min())
    elif method == 'zscore':
        return (data - data.mean()) / data.std()
    else:
        raise ValueError("Invalid normalization method")

def encode_categorical(data, columns, method='onehot'):
    """
    分类数据编码
    可选方法：'onehot'(独热编码), 'label'(标签编码)
    """
    if method == 'onehot':
        return pd.get_dummies(data, columns=columns)
    elif method == 'label':
        for col in columns:
            data[col] = data[col].astype('category').cat.codes
        return data
    else:
        raise ValueError("Invalid encoding method")

def feature_selection(data, target, k=5):
    """
    基于相关性的特征选择
    返回前k个最重要的特征
    """
    corr_matrix = data.corr().abs()
    top_features = corr_matrix[target].sort_values(ascending=False)[1:k+1]
    return data[top_features.index.tolist() + [target]]

def preprocess_pipeline(file_path, target_col=None):
    """
    完整的预处理流程
    """
    print("加载数据...")
    data = load_data(file_path)
    
    print("\n原始数据预览：")
    print(data.head())
    print(f"\n数据形状：{data.shape}")
    
    print("\n处理缺失值...")
    data = handle_missing_values(data, 'mean')
    
    print("删除重复行...")
    data = remove_duplicates(data)
    
    # 分离数值型和分类型列
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    
    print("\n数值型列归一化...")
    if numeric_cols:
        data[numeric_cols] = normalize_data(data[numeric_cols], 'minmax')
    
    print("分类型列编码...")
    if categorical_cols:
        data = encode_categorical(data, categorical_cols, 'onehot')
    
    # 特征选择
    if target_col:
        print(f"\n基于特征选择选择前5个与'{target_col}'最相关的特征...")
        data = feature_selection(data, target_col)
    
    print("\n预处理后数据预览：")
    print(data.head())
    print(f"最终数据形状：{data.shape}")
    
    return data