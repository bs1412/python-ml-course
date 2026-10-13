# main.py
import numpy as np
import pandas as pd
from utils.preprocessing import load_data, handle_missing_values, remove_duplicates, normalize_data, encode_categorical
from utils.visualization import set_plot_style, plot_distribution, plot_box, plot_scatter, plot_correlation, save_plot
from utils.statistical_analysis import describe_data, correlation_analysis, normality_test
from ml_lib.linear_regression import LinearRegression
from ml_lib.knn import KNNClassifier
from ml_lib.decision_tree import DecisionTreeClassifier
from ml_lib.naive_bayes import GaussianNaiveBayes
from ml_lib.kmeans import KMeans

# 设置全局绘图样式
set_plot_style()

# 1. 加载并预处理数据
print("="*50)
print("加载并预处理数据...")
print("="*50)
data = load_data('data/student_habits_performance.csv')
print(f"\n原始数据集形状: {data.shape}")
print("\n数据预览:")
print(data.head())

# 处理缺失值
data = handle_missing_values(data, strategy='mean')
print("\n处理缺失值后数据集形状:", data.shape)

# 删除重复值
data = remove_duplicates(data)
print("\n删除重复值后数据集形状:", data.shape)

# 定义分类列和数值列
categorical_cols = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 
                   'internet_quality', 'extracurricular_participation']
numeric_cols = ['daily_study_time', 'sleep_hours', 'freetime_daily_hours', 
               'social_media_usage', 'exercise_frequency', 'exam_score']

# 编码分类变量
data_encoded = encode_categorical(data[categorical_cols], columns=categorical_cols, method='onehot')
data = pd.concat([data_encoded, data[numeric_cols]], axis=1)

# 归一化数值特征
data_normalized = normalize_data(data, method='zscore')
data_final = pd.DataFrame(data_normalized, columns=data.columns)

print("\n预处理后数据:")
print(data_final.head())

# 2. 数据探索和可视化
print("\n" + "="*50)
print("数据探索和可视化...")
print("="*50)

# 检查考试分数分布
exam_dist = plot_distribution(data, 'exam_score', bins=20, title='考试成绩分布')
save_plot(exam_dist, 'exam_score_distribution')

# 可视化社交媒体使用与学习时间的关系
plt = plot_scatter(data, 'social_media_usage', 'daily_study_time', title='社交媒体使用 vs 学习时间')
save_plot(plt, 'social_media_vs_study_time')

# 检查不同性别的学习时间
gender_study = plot_box(data, x='gender', y='daily_study_time', title='不同性别的学习时间比较')
save_plot(gender_study, 'study_time_by_gender')

# 相关性分析
plt = plot_correlation(data_final.corr(), title='变量相关性热力图')
save_plot(plt, 'correlation_heatmap')

# 3. 统计分析
print("\n" + "="*50)
print("统计分析...")
print("="*50)
stats = describe_data(data_final)
print("\n数值变量统计描述:")
print(stats['numerical'])

print("\n变量相关性矩阵:")
print(correlation_analysis(data_final))

# 检查正态性
print("\n正态性检验结果:")
normality = normality_test(data_final)
for col, res in normality.items():
    normal = "是" if res['is_normal'] else "否"
    print(f"{col}: p值={res['p_value']:.4f}, 正态分布? {normal}")

# 4. 预测建模和比较
print("\n" + "="*50)
print("机器学习模型比较...")
print("="*50)

# 分离特征和目标
X = data_final.drop('exam_score', axis=1).values
y = data['exam_score'].values

# 将成绩分为三档进行分类任务
y_class = np.where(y > 90, '优秀', np.where(y > 80, '良好', '及格'))

# 创建训练测试集
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]
y_train_class, y_test_class = y_class[indices[:split]], y_class[indices[split:]]

def evaluate_regression(model, X_test, y_test):
    """评估回归模型"""
    preds = model.predict(X_test)
    mae = np.mean(np.abs(preds - y_test))
    mse = np.mean((preds - y_test)**2)
    r2 = 1 - np.sum((preds - y_test) ** 2) / np.sum((y_test - np.mean(y_test))**2)
    
    plt = plot_scatter(pd.DataFrame({'实际值': y_test, '预测值': preds}), 
                      '实际值', '预测值', 
                      title=f'{model.__class__.__name__} 预测结果')
    save_plot(plt, f'{model.__class__.__name__}_predictions')
    
    return {
        '算法': model.__class__.__name__,
        'MAE': f"{mae:.4f}",
        'MSE': f"{mse:.4f}",
        'R²': f"{r2:.4f}"
    }

def evaluate_classification(model, X_test, y_test):
    """评估分类模型"""
    preds = model.predict(X_test)
    accuracy = np.mean(preds == y_test)
    
    # 创建混淆矩阵
    classes = np.unique(y_test)
    conf_matrix = pd.DataFrame(
        np.zeros((len(classes), len(classes))), 
        index=classes, columns=classes
    )
    
    for true, pred in zip(y_test, preds):
        conf_matrix.loc[true, pred] += 1
    
    print(f"\n{model.__class__.__name__} 混淆矩阵:")
    print(conf_matrix)
    
    return {
        '算法': model.__class__.__name__,
        '准确率': f"{accuracy:.4f}"
    }

# 训练和评估回归模型
print("\n回归模型评估:")
lr_model = LinearRegression(learning_rate=0.01, n_iterations=1000)
lr_model.fit(X_train, y_train)
reg_results = [evaluate_regression(lr_model, X_test, y_test)]

# 训练和评估分类模型
print("\n分类模型评估:")
# KNN分类器
knn_model = KNNClassifier(k=5)
knn_model.fit(X_train, y_train_class)
knn_results = evaluate_classification(knn_model, X_test, y_test_class)

# 决策树分类器
dt_model = DecisionTreeClassifier(max_depth=5)
dt_model.fit(X_train, y_train_class)
dt_results = evaluate_classification(dt_model, X_test, y_test_class)

# 朴素贝叶斯分类器
nb_model = GaussianNaiveBayes()
nb_model.fit(X_train, y_train_class)
nb_results = evaluate_classification(nb_model, X_test, y_test_class)

# 结果汇总
all_results = reg_results + [
    {'算法': 'KNN', '准确率': knn_results['准确率']},
    {'算法': '决策树', '准确率': dt_results['准确率']},
    {'算法': '朴素贝叶斯', '准确率': nb_results['准确率']}
]

results_df = pd.DataFrame(all_results)
print("\n" + "="*50)
print("模型性能汇总:")
print(results_df)

# 5. 聚类分析
print("\n" + "="*50)
print("聚类分析...")
print("="*50)
# 选择用于聚类的特征
features_for_clustering = ['daily_study_time', 'sleep_hours', 'freetime_daily_hours', 
                          'social_media_usage', 'exercise_frequency']
X_cluster = data_final[features_for_clustering].values

# 应用KMeans聚类
kmeans = KMeans(k=3)
kmeans.fit(X_cluster)
cluster_labels = kmeans.predict(X_cluster)

# 分析不同簇的学习习惯
data['cluster'] = cluster_labels
cluster_summary = data.groupby('cluster')[features_for_clustering].mean()
print("\n各簇的学习习惯平均值:")
print(cluster_summary)

# 可视化聚类结果
plt = plot_scatter(pd.DataFrame({
    '社交使用': data['social_media_usage'], 
    '学习时间': data['daily_study_time'],
    'cluster': cluster_labels
}), '社交使用', '学习时间', hue='cluster', 
title='学习习惯聚类分析 (K=3)')
save_plot(plt, 'learning_habits_clusters')