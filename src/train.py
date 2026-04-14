import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用相对导入
from .data_loader import load_startup_data, explore_data
from .preprocess import preprocess_data
from .logistic_regression import LogisticRegression
from .evaluate import evaluate_model, analyze_feature_importance, plot_confusion_matrix, plot_roc_curve

def train_logistic_regression_model():
    """
    训练逻辑回归模型的完整流程
    """
    print("🚀 开始创业公司盈利预测项目")
    print("=" * 60)
    
    # 1. 数据预处理
    print("\n📊 步骤1: 数据预处理")
    print("-" * 30)
    
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data()
    
    if X_train is None:
        print("❌ 数据预处理失败，程序退出")
        return None
    
    print(f"✅ 预处理完成")
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"特征数量: {len(feature_names)}")
    
    # 2. 模型训练
    print("\n🤖 步骤2: 模型训练")
    print("-" * 30)
    
    # 创建逻辑回归模型
    model = LogisticRegression(
        learning_rate=0.1,
        n_iterations=1000,
        verbose=True
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_history)
    plt.title('逻辑回归训练损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 3. 模型评估
    print("\n📈 步骤3: 模型评估")
    print("-" * 30)
    
    # 基础评估
    results = evaluate_model(model, X_test, y_test, feature_names)
    
    # 特征重要性分析
    importance_df = analyze_feature_importance(model, feature_names)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(results['confusion_matrix'])
    
    # 绘制ROC曲线
    plot_roc_curve(y_test, results['y_pred_proba'])
    
    # 4. 业务洞察
    print("\n💡 步骤4: 业务洞察")
    print("-" * 30)
    
    # 显示最重要的特征
    top_features = importance_df.head(5)
    print("\n🎯 最重要的5个特征:")
    for i, row in top_features.iterrows():
        impact = "促进盈利" if row['权重'] > 0 else "抑制盈利"
        print(f"{i+1}. {row['特征']}: {impact} (重要性: {row['重要性']:.4f})")
    
    # 5. 保存结果
    print("\n💾 步骤5: 保存结果")
    print("-" * 30)
    
    # 确保结果目录存在
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存评估指标
    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"逻辑回归模型评估结果\n")
        f.write(f"=" * 30 + "\n")
        f.write(f"准确率: {results['accuracy']:.4f}\n")
        f.write(f"精确率: {results['precision']:.4f}\n")
        f.write(f"召回率: {results['recall']:.4f}\n")
        f.write(f"F1分数: {results['f1']:.4f}\n")
        f.write(f"AUC-ROC: {results['auc_roc']:.4f}\n")
    
    print(f"✅ 评估指标已保存到: {metrics_path}")
    
    # 保存特征重要性
    importance_path = os.path.join(results_dir, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
    print(f"✅ 特征重要性已保存到: {importance_path}")
    
    print("\n🎉 项目完成!")
    print("=" * 60)
    
    return model, results, importance_df

def interpret_model_results(model, feature_names, top_n=5):
    """
    解释模型结果，提供业务洞察
    """
    print("\n🔍 模型结果解释")
    print("=" * 50)
    
    # 获取特征重要性
    feature_importance = np.abs(model.weights)
    sorted_indices = np.argsort(feature_importance)[::-1]
    
    print("📊 特征对盈利概率的影响:")
    print("-" * 40)
    
    for i in range(min(top_n, len(feature_names))):
        idx = sorted_indices[i]
        feature_name = feature_names[idx]
        weight = model.weights[idx]
        importance = feature_importance[idx]
        
        if weight > 0:
            impact = "📈 促进盈利"
            explanation = "该特征值增加会提高盈利概率"
        else:
            impact = "📉 抑制盈利"
            explanation = "该特征值增加会降低盈利概率"
        
        print(f"{i+1}. {feature_name}")
        print(f"   权重: {weight:.4f} | 重要性: {importance:.4f}")
        print(f"   {impact}: {explanation}")
        
        # 提供业务解释
        if "Revenue" in feature_name:
            print("   💼 业务解释: 收入是盈利的直接驱动力")
        elif "Funding" in feature_name:
            print("   💼 业务解释: 融资效率影响资金使用效果")
        elif "Market" in feature_name:
            print("   💼 业务解释: 市场份额反映市场竞争力")
        elif "Employee" in feature_name:
            print("   💼 业务解释: 人均效率反映运营效率")
        
        print()

def demo_model_predictions(model, X_test, y_test, feature_names, n_examples=5):
    """
    展示模型预测示例
    """
    print("\n🔮 模型预测示例")
    print("=" * 50)
    
    # 随机选择几个样本
    indices = np.random.choice(len(X_test), min(n_examples, len(X_test)), replace=False)
    
    for i, idx in enumerate(indices):
        sample = X_test[idx]
        true_label = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
        
        # 预测概率
        probability = model.predict_proba(sample.reshape(1, -1))[0]
        predicted_label = 1 if probability >= 0.5 else 0
        
        print(f"示例 {i+1}:")
        print(f"   真实标签: {'盈利' if true_label == 1 else '不盈利'}")
        print(f"   预测概率: {probability:.4f}")
        print(f"   预测标签: {'盈利' if predicted_label == 1 else '不盈利'}")
        print(f"   预测结果: {'✅ 正确' if predicted_label == true_label else '❌ 错误'}")
        
        # 显示关键特征值
        print("   关键特征值:")
        important_features = np.argsort(np.abs(model.weights))[::-1][:3]  # 最重要的3个特征
        for feat_idx in important_features:
            feat_name = feature_names[feat_idx]
            feat_value = sample[feat_idx]
            print(f"     {feat_name}: {feat_value:.2f}")
        
        print()

if __name__ == "__main__":
    # 运行完整训练流程
    model, results, importance_df = train_logistic_regression_model()
    
    if model is not None:
        # 加载测试数据用于演示
        from src.preprocess import preprocess_data
        X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data()
        
        # 模型解释
        interpret_model_results(model, feature_names)
        
        # 预测演示
        demo_model_predictions(model, X_test, y_test, feature_names)