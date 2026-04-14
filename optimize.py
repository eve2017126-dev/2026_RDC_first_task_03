#!/usr/bin/env python3
"""
模型优化脚本 - 针对逻辑回归模型进行性能调优
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def optimize_hyperparameters():
    """优化超参数"""
    print("🎯 开始超参数优化...")
    print("=" * 50)
    
    # 加载预处理后的数据
    from src.preprocess import preprocess_data
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data()
    
    if X_train is None:
        print("❌ 数据加载失败")
        return
    
    # 测试不同的学习率和迭代次数组合
    learning_rates = [0.01, 0.05, 0.1, 0.2]
    iterations_list = [500, 1000, 2000]
    
    best_auc = 0
    best_params = {}
    best_model = None
    
    results = []
    
    for lr in learning_rates:
        for n_iter in iterations_list:
            print(f"\n🔧 测试参数: 学习率={lr}, 迭代次数={n_iter}")
            
            # 训练模型
            from src.logistic_regression import LogisticRegression
            model = LogisticRegression(learning_rate=lr, n_iterations=n_iter, verbose=False)
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            
            auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            
            results.append({
                'learning_rate': lr,
                'iterations': n_iter,
                'auc': auc,
                'accuracy': accuracy,
                'final_loss': model.loss_history[-1] if model.loss_history else 0
            })
            
            print(f"   AUC: {auc:.4f}, 准确率: {accuracy:.4f}")
            
            # 更新最佳参数
            if auc > best_auc:
                best_auc = auc
                best_params = {'learning_rate': lr, 'iterations': n_iter}
                best_model = model
    
    # 显示优化结果
    print("\n📊 超参数优化结果:")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('auc', ascending=False)
    print(results_df.to_string(index=False))
    
    print(f"\n🎯 最佳参数: 学习率={best_params['learning_rate']}, 迭代次数={best_params['iterations']}")
    print(f"🏆 最佳AUC: {best_auc:.4f}")
    
    return best_model, best_params, results_df

def optimize_threshold():
    """优化分类阈值"""
    print("\n🎯 开始阈值优化...")
    print("=" * 50)
    
    # 加载数据和模型
    from src.preprocess import preprocess_data
    from src.logistic_regression import LogisticRegression
    
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data()
    
    # 使用最佳参数训练模型
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000, verbose=False)
    model.fit(X_train, y_train)
    
    # 预测概率
    y_pred_proba = model.predict_proba(X_test)
    
    # 测试不同阈值
    thresholds = np.arange(0.3, 0.7, 0.02)
    threshold_results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        threshold_results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # 找到最优阈值
    threshold_df = pd.DataFrame(threshold_results)
    
    # 基于F1分数选择最优阈值
    best_f1_idx = threshold_df['f1'].idxmax()
    best_threshold = threshold_df.loc[best_f1_idx, 'threshold']
    
    print("📊 阈值优化结果:")
    print(threshold_df.round(4).to_string(index=False))
    
    print(f"\n🎯 最优阈值: {best_threshold:.3f}")
    print(f"🏆 最优F1分数: {threshold_df.loc[best_f1_idx, 'f1']:.4f}")
    
    # 绘制阈值优化曲线
    plt.figure(figsize=(12, 8))
    plt.plot(threshold_df['threshold'], threshold_df['accuracy'], label='准确率', marker='o')
    plt.plot(threshold_df['threshold'], threshold_df['precision'], label='精确率', marker='s')
    plt.plot(threshold_df['threshold'], threshold_df['recall'], label='召回率', marker='^')
    plt.plot(threshold_df['threshold'], threshold_df['f1'], label='F1分数', marker='d', linewidth=2)
    
    plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'最优阈值: {best_threshold:.3f}')
    plt.xlabel('分类阈值')
    plt.ylabel('分数')
    plt.title('阈值优化分析')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return best_threshold, threshold_df

def feature_selection_analysis():
    """特征选择分析"""
    print("\n🔍 开始特征选择分析...")
    print("=" * 50)
    
    from src.preprocess import preprocess_data
    from src.logistic_regression import LogisticRegression
    from src.evaluate import analyze_feature_importance
    
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data()
    
    # 训练模型
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000, verbose=False)
    model.fit(X_train, y_train)
    
    # 分析特征重要性
    importance_df = analyze_feature_importance(model, feature_names)
    
    # 特征选择实验
    print("\n🧪 特征选择实验:")
    
    # 测试不同特征数量的效果
    feature_counts = [5, 8, 10, len(feature_names)]
    selection_results = []
    
    for n_features in feature_counts:
        # 选择最重要的n个特征
        top_features = importance_df.head(n_features)['特征'].tolist()
        feature_indices = [feature_names.index(feat) for feat in top_features]
        
        X_train_selected = X_train[:, feature_indices]
        X_test_selected = X_test[:, feature_indices]
        
        # 训练新模型
        model_selected = LogisticRegression(learning_rate=0.1, n_iterations=1000, verbose=False)
        model_selected.fit(X_train_selected, y_train)
        
        # 评估
        y_pred_proba = model_selected.predict_proba(X_test_selected)
        y_pred = model_selected.predict(X_test_selected)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        
        selection_results.append({
            'n_features': n_features,
            'auc': auc,
            'accuracy': accuracy,
            'features': ', '.join(top_features[:3]) + '...' if n_features > 3 else ', '.join(top_features)
        })
        
        print(f"   特征数量: {n_features:2d} | AUC: {auc:.4f} | 准确率: {accuracy:.4f}")
    
    selection_df = pd.DataFrame(selection_results)
    print("\n📊 特征选择结果:")
    print(selection_df.to_string(index=False))
    
    return selection_df, importance_df

def generate_optimization_report():
    """生成优化报告"""
    print("🚀 开始全面模型优化")
    print("=" * 60)
    
    # 1. 超参数优化
    best_model, best_params, hyper_results = optimize_hyperparameters()
    
    # 2. 阈值优化
    best_threshold, threshold_results = optimize_threshold()
    
    # 3. 特征选择分析
    selection_results, importance_df = feature_selection_analysis()
    
    # 生成优化报告
    report_content = f"""
逻辑回归模型优化报告
====================

最佳超参数:
- 学习率: {best_params['learning_rate']}
- 迭代次数: {best_params['iterations']}
- 最佳AUC: {hyper_results['auc'].max():.4f}

最佳分类阈值:
- 阈值: {best_threshold:.3f}
- 对应F1分数: {threshold_results['f1'].max():.4f}

特征选择建议:
- 推荐特征数量: {selection_results.loc[selection_results['auc'].idxmax(), 'n_features']}
- 对应AUC: {selection_results['auc'].max():.4f}

优化建议:
1. 使用最佳超参数重新训练模型
2. 调整分类阈值为 {best_threshold:.3f}
3. 考虑特征选择以提升模型性能
"""
    
    # 保存报告
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "optimization_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n✅ 优化报告已保存到: {report_path}")
    print("🎉 模型优化完成!")
    
    return report_content

if __name__ == "__main__":
    generate_optimization_report()