import pandas as pd
import numpy as np
from scipy import stats
import itertools

# 直接导入预处理模块（不再使用相对导入）
# 因为统计分析模块和预处理模块在同一个utils目录
import utils.preprocessing as dp

def describe_data(data):
    """生成数据的描述性统计信息"""
    result = {}
    
    # 数值型变量的描述性统计
    num_desc = data.select_dtypes(include=np.number).describe().round(2)
    result['numerical'] = num_desc
    
    # 分类型变量的统计
    cat_desc = []
    for col in data.select_dtypes(exclude=np.number).columns:
        counts = data[col].value_counts()
        percents = data[col].value_counts(normalize=True).mul(100).round(1)
        cat_desc.append(pd.concat([counts, percents], axis=1, 
                                  keys=[f'{col}_count', f'{col}_%']))
    result['categorical'] = cat_desc
    
    return result

def correlation_analysis(data, method='pearson'):
    """
    分析变量间的相关性
    - method: 'pearson'(默认), 'spearman', 'kendall'
    """
    # 只选择数值型列进行计算
    numeric_data = data.select_dtypes(include=np.number)
    if numeric_data.empty:
        print("警告: 没有数值型列可用于相关性分析")
        return pd.DataFrame()
        
    return numeric_data.corr(method=method).round(3)

def outlier_detection(data, method='iqr', threshold=1.5):
    """
    检测数值型变量的异常值
    - method: 'iqr'(IQR法), 'zscore'(Z分数法)
    - threshold: 阈值
    """
    outliers = {}
    for col in data.select_dtypes(include=np.number).columns:
        if method == 'iqr':
            # IQR方法检测异常值
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            col_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        elif method == 'zscore':
            # Z分数方法检测异常值
            z_scores = np.abs(stats.zscore(data[col]))
            col_outliers = data[z_scores > threshold]
        
        if not col_outliers.empty:
            outliers[col] = {
                'count': len(col_outliers),
                'samples': col_outliers
            }
    
    return outliers

def normality_test(data):
    """
    检验数值型变量的正态性
    - 使用Shapiro-Wilk检验
    """
    normality_results = {}
    for col in data.select_dtypes(include=np.number).columns:
        stat, p = stats.shapiro(data[col].dropna())
        normality_results[col] = {
            'statistic': round(stat, 4),
            'p_value': round(p, 4),
            'is_normal': p > 0.05
        }
    return normality_results

def contingency_analysis(data, category_col1, category_col2):
    """
    两个分类型变量的列联表和卡方检验
    """
    # 创建列联表
    contingency_table = pd.crosstab(data[category_col1], data[category_col2])
    
    # 执行卡方检验
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    
    return {
        'contingency_table': contingency_table,
        'chi2_statistic': round(chi2, 4),
        'p_value': round(p, 4),
        'dof': dof,
        'expected_freq': pd.DataFrame(expected, 
                                      index=contingency_table.index, 
                                      columns=contingency_table.columns)
    }

def t_test(data, group_col, value_col):
    """
    独立样本T检验
    - 比较两组在不同类别下数值型变量的均值是否有显著差异
    """
    groups = data[group_col].unique()
    if len(groups) != 2:
        raise ValueError("分组变量必须恰好有2个类别")
    
    group1 = data[data[group_col] == groups[0]][value_col].dropna()
    group2 = data[data[group_col] == groups[1]][value_col].dropna()
    
    # 执行T检验
    stat, p = stats.ttest_ind(group1, group2, nan_policy='omit')
    
    return {
        'groups': groups,
        'mean_group1': round(group1.mean(), 4),
        'mean_group2': round(group2.mean(), 4),
        't_statistic': round(stat, 4),
        'p_value': round(p, 4)
    }

def anova_test(data, group_col, value_col):
    """
    方差分析(ANOVA)
    - 比较三组及以上不同类别下数值型变量的均值是否有显著差异
    """
    groups = []
    for category in data[group_col].unique():
        groups.append(data[data[group_col] == category][value_col].dropna())
    
    # 执行ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    return {
        'f_statistic': round(f_stat, 4),
        'p_value': round(p_value, 4),
    }

def correlation_summary(data):
    """生成简单的相关性摘要"""
    # 只使用数值列进行计算
    corr = correlation_analysis(data)
    
    if corr.empty:
        print("无法计算相关性 - 缺少数值列")
        return []
    
    # 找出相关性最高的三对变量
    max_corr = []
    for col in corr.columns:
        # 忽略与自身的相关性
        sorted_corr = corr[col].drop(col).sort_values(ascending=False)
        if not sorted_corr.empty:
            max_val = sorted_corr.iloc[0]
            max_idx = sorted_corr.index[0]
            max_corr.append((col, max_idx, max_val))
    
    # 按相关性从高到低排序
    max_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return max_corr[:3]  # 返回前三对

def analysis_pipeline(data_path, target_col=None):
    """
    简化的统计分析流程（适合大一新生作业）
    """
    print("\n" + "="*50)
    print(f"分析数据集: {data_path}")
    
    # 使用预处理模块加载数据
    data = dp.load_data(data_path)
    
    # 简单预处理
    print("处理缺失值...")
   # 只对数值列使用均值填充，对字符串列使用众数填充
    numeric_cols = data.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
           data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
   
    categorical_cols = data.select_dtypes(exclude=np.number).columns
    if not categorical_cols.empty:
       # 对分类列使用众数填充
           data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
   
    print("删除重复行...")
    data = dp.remove_duplicates(data)
    
    # 1. 基本统计信息
    num_cols = data.select_dtypes(include=np.number).columns
    cat_cols = data.select_dtypes(exclude=np.number).columns
    
    print(f"\n数据集包含 {len(data)} 行，{len(data.columns)} 列")
    print(f"数值型列: {', '.join(num_cols)}")
    print(f"分类型列: {', '.join(cat_cols)}")
    
    # 2. 描述性统计摘要
    print("\n描述性统计摘要:")
    desc = describe_data(data)
    
    # 打印数值型变量摘要
    for col in num_cols:
        print(f"\n变量: {col}")
        col_stats = desc['numerical'][col]
        print(f"平均值: {col_stats['mean']:.2f} | 中位数: {col_stats['50%']:.2f}")
        print(f"最小值: {col_stats['min']:.2f} | 最大值: {col_stats['max']:.2f}")
        print(f"标准差: {col_stats['std']:.2f}")
    
    # 3. 相关性分析
    if len(num_cols) >= 2:
        top_correlations = correlation_summary(data)
        print("\n相关性最强的变量对:")
        for col1, col2, corr_value in top_correlations:
            print(f"{col1} 和 {col2}: {corr_value:.2f}")
    
    # 4. 分组比较（如果指定分组变量）
    if cat_cols.any():
        group_col = cat_cols[0]  # 使用第一个分类变量作为分组依据
        
        if len(num_cols) > 0:
            num_col = num_cols[0]  # 使用第一个数值变量进行比较
            
            print(f"\n基于 '{group_col}' 分组的 '{num_col}' 比较:")
            
            # 如果是两组，用T检验
            if len(data[group_col].unique()) == 2:
                t_result = t_test(data, group_col, num_col)
                print(f"组1({t_result['groups'][0]})均值: {t_result['mean_group1']:.2f}")
                print(f"组2({t_result['groups'][1]})均值: {t_result['mean_group2']:.2f}")
                print(f"T检验p值: {t_result['p_value']:.4f}")
            
            # 如果是三组或以上，用ANOVA
            elif len(data[group_col].unique()) >= 3:
                anova_result = anova_test(data, group_col, num_col)
                print(f"ANOVA p值: {anova_result['p_value']:.4f}")
    
    # 5. 目标变量分析（如果指定）
    if target_col and target_col in data.columns:
        print(f"\n目标变量 '{target_col}' 分析:")
        
        # 如果是数值型目标变量
        if data[target_col].dtype.kind in 'fiu':
            print(f"平均值: {data[target_col].mean():.2f}")
            print(f"最小值: {data[target_col].min():.2f} | 最大值: {data[target_col].max():.2f}")
        
        # 如果是分类型目标变量
        else:
            counts = data[target_col].value_counts()
            print("分类分布:")
            for val, count in counts.items():
                percent = count / len(data) * 100
                print(f"{val}: {count} ({percent:.1f}%)")
    
    print("\n" + "="*50)
    
    # 返回处理后的数据用于进一步分析
    return data