#!/usr/bin/env python3
"""
测试脚本 - 验证项目能否正常运行
"""

import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有模块能否正常导入"""
    print("🧪 开始测试模块导入...")
    
    modules_to_test = [
        ('data_loader', 'load_startup_data'),
        ('preprocess', 'preprocess_data'),
        ('logistic_regression', 'LogisticRegression'),
        ('evaluate', 'evaluate_model'),
        ('train', 'train_logistic_regression_model')
    ]
    
    for module_name, function_name in modules_to_test:
        try:
            # 尝试从src目录导入
            exec(f"from src.{module_name} import {function_name}")
            print(f"✅ src.{module_name} 导入成功")
        except ImportError as e:
            print(f"❌ src.{module_name} 导入失败: {e}")
            
            try:
                # 尝试直接导入
                exec(f"from {module_name} import {function_name}")
                print(f"✅ {module_name} 直接导入成功")
            except ImportError as e2:
                print(f"❌ {module_name} 直接导入失败: {e2}")

def test_data_loading():
    """测试数据加载功能"""
    print("\n📊 测试数据加载...")
    
    try:
        from src.data_loader import load_startup_data
        data = load_startup_data()
        if data is not None:
            print(f"✅ 数据加载成功，形状: {data.shape}")
            print(f"   列名: {list(data.columns)}")
        else:
            print("❌ 数据加载失败")
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")

def test_preprocessing():
    """测试数据预处理功能"""
    print("\n🔧 测试数据预处理...")
    
    try:
        from src.preprocess import preprocess_data
        X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data()
        
        if X_train is not None:
            print(f"✅ 数据预处理成功")
            print(f"   训练集形状: {X_train.shape}")
            print(f"   测试集形状: {X_test.shape}")
            print(f"   特征数量: {len(feature_names)}")
        else:
            print("❌ 数据预处理失败")
    except Exception as e:
        print(f"❌ 数据预处理测试失败: {e}")

def main():
    """主测试函数"""
    print("🚀 开始项目测试")
    print("=" * 50)
    
    # 测试模块导入
    test_imports()
    
    # 测试数据加载
    test_data_loading()
    
    # 测试数据预处理
    test_preprocessing()
    
    print("\n" + "=" * 50)
    print("📋 测试完成")

if __name__ == "__main__":
    main()