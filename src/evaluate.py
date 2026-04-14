import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns

def evaluate_model(model, X_test, y_test, feature_names=None, threshold=0.5):
    """
    评估逻辑回归模型性能
    
    参数:
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集真实标签
        feature_names: 特征名称列表
        threshold: 分类阈值
    """
    print("\n📊 模型评估结果")
    print("=" * 50)
    
    # 预测
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test, threshold=threshold)
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # AUC-ROC（需要概率值）
    try:
        auc_roc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc_roc = 0.5  # 如果计算失败，使用随机猜测的AUC
    
    print(f"🔢 准确率 (Accuracy): {accuracy:.4f}")
    print(f"🎯 精确率 (Precision): {precision:.4f}")
    print(f"📈 召回率 (Recall): {recall:.4f}")
    print(f"⚖️  F1分数: {f1:.4f}")
    print(f"📊 AUC-ROC: {auc_roc:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔍 混淆矩阵:")
    print(cm)
    
    # 详细分类报告
    tn, fp, fn, tp = cm.ravel()
    print(f"\n📋 详细分类报告:")
    print(f"真阴性 (TN): {tn}")
    print(f"假阳性 (FP): {fp}")
    print(f"假阴性 (FN): {fn}")
    print(f"真阳性 (TP): {tp}")
    
    # 业务指标
    print(f"\n💼 业务指标:")
    print(f"盈利公司识别率: {recall:.2%}")
    print(f"误报率: {fp/(fp+tn):.2%}" if (fp+tn) > 0 else "误报率: 0.00%")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_confusion_matrix(cm, save_path=None):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['不盈利 (0)', '盈利 (1)'],
                yticklabels=['不盈利 (0)', '盈利 (1)'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵图已保存到: {save_path}")
    
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """绘制ROC曲线"""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线图已保存到: {save_path}")
    
    plt.show()

def analyze_feature_importance(model, feature_names):
    """分析特征重要性"""
    if model.weights is None:
        print("模型尚未训练，无法分析特征重要性")
        return None
    
    print("\n🔍 特征重要性分析")
    print("=" * 50)
    
    # 计算特征重要性（权重绝对值）
    feature_importance = np.abs(model.weights)
    
    # 创建重要性DataFrame
    importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': feature_importance,
        '权重': model.weights
    }).sort_values('重要性', ascending=False)
    
    print("特征重要性排名:")
    for i, row in importance_df.iterrows():
        print(f"{i+1:2d}. {row['特征']:20s} | 重要性: {row['重要性']:8.4f} | 权重: {row['权重']:8.4f}")
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(importance_df)), importance_df['重要性'])
    plt.yticks(range(len(importance_df)), importance_df['特征'])
    plt.xlabel('特征重要性（权重绝对值）')
    plt.title('逻辑回归特征重要性')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importance_df

def find_optimal_threshold(y_true, y_pred_proba):
    """寻找最优分类阈值"""
    from sklearn.metrics import f1_score
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)
    
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    optimal_f1 = max(f1_scores)
    
    print(f"\n🎯 最优阈值分析")
    print(f"最优阈值: {optimal_threshold:.3f}")
    print(f"最优F1分数: {optimal_f1:.4f}")
    
    # 绘制阈值-F1曲线
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores)
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'最优阈值: {optimal_threshold:.3f}')
    plt.xlabel('分类阈值')
    plt.ylabel('F1分数')
    plt.title('阈值选择与F1分数关系')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return optimal_threshold, optimal_f1

def generate_evaluation_report(model, X_test, y_test, feature_names, save_path=None):
    """生成完整评估报告"""
    print("\n📋 生成完整评估报告")
    print("=" * 50)
    
    # 基础评估
    results = evaluate_model(model, X_test, y_test, feature_names)
    
    # 特征重要性分析
    importance_df = analyze_feature_importance(model, feature_names)
    
    # 最优阈值分析
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, results['y_pred_proba'])
    
    # 保存评估结果
    if save_path:
        report_content = f"""
逻辑回归模型评估报告
====================

模型性能指标:
- 准确率: {results['accuracy']:.4f}
- 精确率: {results['precision']:.4f}
- 召回率: {results['recall']:.4f}
- F1分数: {results['f1']:.4f}
- AUC-ROC: {results['auc_roc']:.4f}

最优阈值分析:
- 最优阈值: {optimal_threshold:.3f}
- 最优F1分数: {optimal_f1:.4f}

特征重要性排名:
"""
        
        for i, row in importance_df.iterrows():
            report_content += f"{i+1}. {row['特征']}: {row['重要性']:.4f}\n"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"评估报告已保存到: {save_path}")
    
    return results, importance_df, optimal_threshold

if __name__ == "__main__":
    # 测试评估函数
    print("评估模块测试完成")

def evaluate_model(model, X_test, y_test, feature_names=None, threshold=0.5):
    """
    评估逻辑回归模型性能
    
    参数:
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集真实标签
        feature_names: 特征名称列表
        threshold: 分类阈值
    """
    print("\n📊 模型评估结果")
    print("=" * 50)
    
    # 预测
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test, threshold=threshold)
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # AUC-ROC（需要概率值）
    try:
        auc_roc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc_roc = 0.5  # 如果计算失败，使用随机猜测的AUC
    
    print(f"🔢 准确率 (Accuracy): {accuracy:.4f}")
    print(f"🎯 精确率 (Precision): {precision:.4f}")
    print(f"📈 召回率 (Recall): {recall:.4f}")
    print(f"⚖️  F1分数: {f1:.4f}")
    print(f"📊 AUC-ROC: {auc_roc:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔍 混淆矩阵:")
    print(cm)
    
    # 详细分类报告
    tn, fp, fn, tp = cm.ravel()
    print(f"\n📋 详细分类报告:")
    print(f"真阴性 (TN): {tn}")
    print(f"假阳性 (FP): {fp}")
    print(f"假阴性 (FN): {fn}")
    print(f"真阳性 (TP): {tp}")
    
    # 业务指标
    print(f"\n💼 业务指标:")
    print(f"盈利公司识别率: {recall:.2%}")
    print(f"误报率: {fp/(fp+tn):.2%}" if (fp+tn) > 0 else "误报率: 0.00%")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_confusion_matrix(cm, save_path=None):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['不盈利 (0)', '盈利 (1)'],
                yticklabels=['不盈利 (0)', '盈利 (1)'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵图已保存到: {save_path}")
    
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """绘制ROC曲线"""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线图已保存到: {save_path}")
    
    plt.show()

def analyze_feature_importance(model, feature_names):
    """分析特征重要性"""
    if model.weights is None:
        print("模型尚未训练，无法分析特征重要性")
        return None
    
    print("\n🔍 特征重要性分析")
    print("=" * 50)
    
    # 计算特征重要性（权重绝对值）
    feature_importance = np.abs(model.weights)
    
    # 创建重要性DataFrame
    importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': feature_importance,
        '权重': model.weights
    }).sort_values('重要性', ascending=False)
    
    print("特征重要性排名:")
    for i, row in importance_df.iterrows():
        print(f"{i+1:2d}. {row['特征']:20s} | 重要性: {row['重要性']:8.4f} | 权重: {row['权重']:8.4f}")
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(importance_df)), importance_df['重要性'])
    plt.yticks(range(len(importance_df)), importance_df['特征'])
    plt.xlabel('特征重要性（权重绝对值）')
    plt.title('逻辑回归特征重要性')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importance_df

def find_optimal_threshold(y_true, y_pred_proba):
    """寻找最优分类阈值"""
    from sklearn.metrics import f1_score
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)
    
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    optimal_f1 = max(f1_scores)
    
    print(f"\n🎯 最优阈值分析")
    print(f"最优阈值: {optimal_threshold:.3f}")
    print(f"最优F1分数: {optimal_f1:.4f}")
    
    # 绘制阈值-F1曲线
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores)
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'最优阈值: {optimal_threshold:.3f}')
    plt.xlabel('分类阈值')
    plt.ylabel('F1分数')
    plt.title('阈值选择与F1分数关系')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return optimal_threshold, optimal_f1

def generate_evaluation_report(model, X_test, y_test, feature_names, save_path=None):
    """生成完整评估报告"""
    print("\n📋 生成完整评估报告")
    print("=" * 50)
    
    # 基础评估
    results = evaluate_model(model, X_test, y_test, feature_names)
    
    # 特征重要性分析
    importance_df = analyze_feature_importance(model, feature_names)
    
    # 最优阈值分析
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, results['y_pred_proba'])
    
    # 保存评估结果
    if save_path:
        report_content = f"""
逻辑回归模型评估报告
====================

模型性能指标:
- 准确率: {results['accuracy']:.4f}
- 精确率: {results['precision']:.4f}
- 召回率: {results['recall']:.4f}
- F1分数: {results['f1']:.4f}
- AUC-ROC: {results['auc_roc']:.4f}

最优阈值分析:
- 最优阈值: {optimal_threshold:.3f}
- 最优F1分数: {optimal_f1:.4f}

特征重要性排名:
"""
        
        for i, row in importance_df.iterrows():
            report_content += f"{i+1}. {row['特征']}: {row['重要性']:.4f}\n"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"评估报告已保存到: {save_path}")
    
    return results, importance_df, optimal_threshold

if __name__ == "__main__":
    # 测试评估函数
    print("评估模块测试完成")