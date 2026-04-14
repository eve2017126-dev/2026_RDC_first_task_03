#!/usr/bin/env python3
"""
创业公司盈利预测 - 逻辑回归模型项目

项目概述:
- 使用手写逻辑回归模型预测创业公司是否盈利
- 重点考察数据预处理、特征工程和模型实现
- 提供完整的评估和业务洞察

任务要求:
1. 数据集分割 (3%)
2. 数据预处理和特征工程 (7%)
3. 手写逻辑回归算法 (7%)
4. 模型评估 (3%)
5. 数学原理理解和答辩 (20%)
"""

import os
import sys
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='创业公司盈利预测项目')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'demo', 'full'], 
                       default='full', help='运行模式')
    parser.add_argument('--data-path', type=str, help='数据文件路径')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='学习率')
    parser.add_argument('--iterations', type=int, default=1000, help='迭代次数')
    
    args = parser.parse_args()
    
    print("🚀 创业公司盈利预测项目")
    print("=" * 60)
    
    try:
        if args.mode in ['train', 'full']:
            try:
                from src.train import train_logistic_regression_model
            except ImportError:
                # 如果src导入失败，尝试直接导入
                from train import train_logistic_regression_model
            
            print("\n📊 开始训练逻辑回归模型...")
            model, results, importance_df = train_logistic_regression_model()
            
            if model is None:
                print("❌ 训练失败")
                return
        
        if args.mode in ['evaluate', 'full'] and 'model' in locals():
            from src.evaluate import generate_evaluation_report
            from src.preprocess import preprocess_data
            
            print("\n📈 开始模型评估...")
            X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(args.data_path)
            
            if X_test is not None:
                # 生成完整评估报告
                results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
                os.makedirs(results_dir, exist_ok=True)
                
                report_path = os.path.join(results_dir, "evaluation_report.txt")
                generate_evaluation_report(model, X_test, y_test, feature_names, report_path)
        
        if args.mode in ['demo', 'full'] and 'model' in locals():
            from src.train import interpret_model_results, demo_model_predictions
            from src.preprocess import preprocess_data
            
            print("\n🔍 开始模型演示...")
            X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(args.data_path)
            
            if X_test is not None:
                # 模型解释
                interpret_model_results(model, feature_names)
                
                # 预测演示
                demo_model_predictions(model, X_test, y_test, feature_names)
        
        print("\n✅ 项目执行完成!")
        
        # 显示下一步建议
        print("\n📋 下一步建议:")
        print("1. 查看 results/ 目录下的评估结果")
        print("2. 分析特征重要性，理解业务逻辑")
        print("3. 准备数学原理笔记和答辩材料")
        print("4. 尝试调整超参数优化模型性能")
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

def quick_start():
    """快速开始指南"""
    print("🚀 快速开始指南")
    print("=" * 50)
    print("\n1. 基础运行:")
    print("   python main.py")
    
    print("\n2. 只训练模型:")
    print("   python main.py --mode train")
    
    print("\n3. 只进行评估:")
    print("   python main.py --mode evaluate")
    
    print("\n4. 自定义参数:")
    print("   python main.py --learning-rate 0.05 --iterations 2000")
    
    print("\n5. 使用自定义数据:")
    print("   python main.py --data-path /path/to/your/data.csv")

if __name__ == "__main__":
    # 如果没有参数，显示快速开始指南
    if len(sys.argv) == 1:
        quick_start()
        print("\n💡 提示: 运行 'python main.py --mode full' 开始完整流程")
    else:
        main()