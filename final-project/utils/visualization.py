import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局绘图样式
def set_plot_style(figsize=(10, 6)):
    """
    设置全局绘图样式
    """
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    print("绘图样式已设置")

# 保存图表
def save_plot(plt, filename, dpi=300, format='png'):
    """
    保存图表
    """
    plt.tight_layout()
    plt.savefig(f"{filename}.{format}", dpi=dpi, bbox_inches='tight')
    print(f"图表已保存为 {filename}.{format}")

# 基本分布图
def plot_distribution(data, column, kde=True, bins=30, title=None):
    """
    绘制单一变量的分布图
    """
    plt.figure()
    sns.histplot(data[column], kde=kde, bins=bins)
    plt.title(title or f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    return plt

# 箱线图
def plot_box(data, x=None, y=None, title=None):
    """
    绘制箱线图
    """
    plt.figure()
    sns.boxplot(data=data, x=x, y=y)
    plt.title(title or f'Boxplot: {x} vs {y}')
    plt.xticks(rotation=45)
    return plt

# 散点图
def plot_scatter(data, x, y, hue=None, title=None):
    """
    绘制散点图
    """
    plt.figure()
    sns.scatterplot(data=data, x=x, y=y, hue=hue)
    plt.title(title or f'{x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    if hue:
        plt.legend(title=hue)
    return plt

# 相关热力图
def plot_correlation(data, title=None):
    """
    绘制变量相关性热力图
    """
    plt.figure()
    corr = data.corr(numeric_only=True)
    
    # 创建上三角掩码
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # 绘制热力图
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                center=0, vmin=-1, vmax=1, linewidths=0.5)
    
    plt.title(title or 'Correlation Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    return plt

# 计数图
def plot_count(data, column, hue=None, title=None):
    """
    绘制分类变量计数图
    """
    plt.figure()
    sns.countplot(data=data, x=column, hue=hue)
    plt.title(title or f'Count of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # 在柱状图上显示计数
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', 
                   xytext=(0, 5), 
                   textcoords='offset points')
    
    return plt

# 时间序列图
def plot_timeseries(data, date_column, value_column, title=None):
    """
    绘制时间序列图
    """
    plt.figure()
    # 确保日期列是日期类型
    if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column])
    
    # 按日期排序
    data = data.sort_values(date_column)
    
    # 绘制折线图
    plt.plot(data[date_column], data[value_column])
    plt.title(title or f'Time Series: {value_column} over time')
    plt.xlabel(date_column)
    plt.ylabel(value_column)
    plt.xticks(rotation=45)
    return plt

# 成对关系图
def plot_pairwise(data, columns=None, hue=None, title=None):
    """
    绘制变量间成对关系图
    """
    if not columns:
        columns = data.select_dtypes(include=np.number).columns.tolist()
    
    if len(columns) > 5:
        print("警告：选择的列较多，可能会生成大量图表")
    
    plot = sns.pairplot(data[columns], hue=hue, diag_kind='kde')
    plot.fig.suptitle(title or 'Pairwise Relationships', y=1.02)
    return plot.fig

# 回归图
def plot_regression(data, x, y, title=None):
    """
    绘制回归关系图
    """
    plt.figure()
    sns.regplot(data=data, x=x, y=y, scatter_kws={'alpha': 0.5})
    plt.title(title or f'Regression: {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    return plt

# 快速概览函数
def quick_overview(data, num_cols=None, cat_cols=None):
    """
    对数据集进行快速可视化概览
    """
    if not num_cols:
        num_cols = data.select_dtypes(include=np.number).columns.tolist()
    
    if not cat_cols:
        cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    
    # 创建4个子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 数值变量的分布
    if num_cols:
        sns.histplot(data[num_cols[0]], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title(f'Distribution of {num_cols[0]}')
    
    # 数值变量的箱线图
    if num_cols:
        sns.boxplot(data=data, y=num_cols[0], ax=axes[0, 1])
        axes[0, 1].set_title(f'Boxplot of {num_cols[0]}')
    
    # 分类变量的计数图
    if cat_cols:
        sns.countplot(data=data, x=cat_cols[0], ax=axes[1, 0])
        axes[1, 0].set_title(f'Count of {cat_cols[0]}')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 相关矩阵
    if num_cols:
        corr = data[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                   center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Correlation Matrix')
    
    plt.tight_layout()
    return fig