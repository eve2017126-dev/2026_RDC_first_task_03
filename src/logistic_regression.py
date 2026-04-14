import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    手写逻辑回归模型
    
    数学原理:
    1. 假设函数: h(x) = sigmoid(θ^T * x) = 1 / (1 + e^(-θ^T * x))
    2. 损失函数: J(θ) = -1/m * Σ [y*log(h(x)) + (1-y)*log(1-h(x))]
    3. 梯度下降: θ_j := θ_j - α * ∂J(θ)/∂θ_j
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=True):
        """
        初始化模型参数
        
        参数:
            learning_rate: 学习率
            n_iterations: 迭代次数
            verbose: 是否显示训练过程
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def sigmoid(self, z):
        """
        Sigmoid激活函数
        
        数学公式: σ(z) = 1 / (1 + e^(-z))
        
        参数:
            z: 线性组合结果
            
        返回:
            sigmoid转换后的概率值
        """
        # 防止数值溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y_true, y_pred):
        """
        计算逻辑回归损失函数（对数损失）
        
        数学公式: J(θ) = -1/m * Σ [y*log(h(x)) + (1-y)*log(1-h(x))]
        
        参数:
            y_true: 真实标签
            y_pred: 预测概率
            
        返回:
            损失值
        """
        # 防止log(0)的情况
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        m = len(y_true)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def fit(self, X, y):
        """
        训练逻辑回归模型
        
        参数:
            X: 特征矩阵 (m samples, n features)
            y: 目标变量 (m samples,)
        """
        m, n = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n)
        self.bias = 0
        
        print("🚀 开始训练逻辑回归模型...")
        print(f"样本数量: {m}, 特征数量: {n}")
        print(f"学习率: {self.learning_rate}, 迭代次数: {self.n_iterations}")
        
        for i in range(self.n_iterations):
            # 前向传播
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # 计算损失
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # 计算梯度（反向传播）
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 每100次迭代显示进度
            if self.verbose and i % 100 == 0:
                print(f"迭代 {i}/{self.n_iterations}, 损失: {loss:.4f}")
        
        print(f"✅ 训练完成! 最终损失: {loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """
        预测概率值
        
        参数:
            X: 特征矩阵
            
        返回:
            预测概率 (0到1之间)
        """
        if self.weights is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """
        预测类别
        
        参数:
            X: 特征矩阵
            threshold: 分类阈值 (默认0.5)
            
        返回:
            预测类别 (0或1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def get_parameters(self):
        """获取模型参数"""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'feature_importance': np.abs(self.weights)
        }
    
    def plot_loss_curve(self, save_path=None):
        """绘制损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.xlabel('迭代次数')
        plt.ylabel('损失值')
        plt.title('逻辑回归训练损失曲线')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"损失曲线已保存到: {save_path}")
        
        plt.show()

def test_logistic_regression():
    """测试逻辑回归模型"""
    print("🧪 测试逻辑回归模型...")
    
    # 创建测试数据
    np.random.seed(42)
    X_test = np.random.randn(100, 3)
    weights_true = np.array([2.0, -1.0, 0.5])
    bias_true = 1.0
    
    # 生成真实概率
    linear_combination = np.dot(X_test, weights_true) + bias_true
    probabilities_true = 1 / (1 + np.exp(-linear_combination))
    
    # 生成标签
    y_test = (probabilities_true > 0.5).astype(int)
    
    # 训练模型
    model = LogisticRegression(learning_rate=0.1, n_iterations=500, verbose=False)
    model.fit(X_test, y_test)
    
    # 预测
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"测试准确率: {accuracy:.4f}")
    print(f"模型权重: {model.weights}")
    print(f"模型偏置: {model.bias}")
    
    return model

if __name__ == "__main__":
    # 测试模型
    model = test_logistic_regression()