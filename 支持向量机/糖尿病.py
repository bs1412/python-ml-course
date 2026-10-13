import numpy as np
import pandas as pd

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
       
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output).astype(int)

    def decision_function(self, X):
        return np.dot(X, self.w) - self.b
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == np.where(y <= 0, -1, 1))

def preprocess_data(df):
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_features:
        median_value = df[col].median()
        df[col] = df[col].replace(0, median_value)
    
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values
    
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    return X, y

if __name__ == "__main__":
    try:
        df = pd.read_csv('diabetes.csv')
    except:
        print("错误：找不到文件'diabetes.csv'")
        exit()
    
    X, y = preprocess_data(df)
    
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split = int(0.8 * n_samples)
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]
    
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_train, y_train)
    
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_test, y_test)
    
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    
    print("\n部分测试样本预测结果:")
    for i in range(3):
        decision_value = svm.decision_function([X_test[i]])[0]
        prediction = "糖尿病" if svm.predict([X_test[i]])[0] == 1 else "健康"
        actual = "糖尿病" if y_test[i] == 1 else "健康"
        
        print(f"样本 {i+1}:")
        print(f"  决策值: {decision_value:.4f}")
        print(f"  预测: {prediction}")
        print(f"  实际: {actual}")
        print()