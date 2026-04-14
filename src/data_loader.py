import pandas as pd
import numpy as np
import os

def load_startup_data(data_path=None):
    """
    加载创业公司数据集
    
    参数:
        data_path: 数据文件路径，如果为None则使用默认路径
        
    返回:
        data: 加载的DataFrame
    """
    if data_path is None:
        # 使用默认路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_path = os.path.join(project_root, "一轮数据集", "startup_data.csv")
    
    print(f"📂 加载数据文件: {data_path}")
    
    try:
        data = pd.read_csv(data_path)
        print(f"✅ 数据加载成功")
        print(f"📊 数据集大小: {data.shape}")
        print(f"🔢 特征数量: {data.shape[1] - 1}")
        print(f"🎯 目标变量: Profitable (1=盈利, 0=不盈利)")
        
        # 显示数据基本信息
        print("\n📋 数据基本信息:")
        print(data.info())
        
        # 显示目标变量分布
        print("\n📈 目标变量分布:")
        profitable_counts = data['Profitable'].value_counts()
        print(profitable_counts)
        print(f"盈利比例: {profitable_counts[1] / len(data) * 100:.2f}%")
        
        return data
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def explore_data(data):
    """
    探索性数据分析
    
    参数:
        data: 加载的数据集
    """
    print("\n🔍 探索性数据分析")
    print("=" * 50)
    
    # 数值特征统计
    print("\n📊 数值特征统计:")
    numeric_cols = ['Funding Rounds', 'Funding Amount (M USD)', 'Valuation (M USD)', 
                   'Revenue (M USD)', 'Employees', 'Market Share (%)']
    print(data[numeric_cols].describe())
    
    # 分类特征统计
    print("\n🏷️ 分类特征统计:")
    print("行业分布:")
    print(data['Industry'].value_counts())
    
    # 缺失值检查
    print("\n❓ 缺失值检查:")
    missing_data = data.isnull().sum()
    print(missing_data[missing_data > 0])
    
    # 相关性分析
    print("\n📈 特征与盈利的相关性:")
    numeric_cols_with_target = numeric_cols + ['Profitable']
    correlation = data[numeric_cols_with_target].corr()['Profitable'].sort_values(ascending=False)
    print(correlation)

def save_processed_data(data, save_path=None):
    """
    保存处理后的数据
    
    参数:
        data: 处理后的数据
        save_path: 保存路径
    """
    if save_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        save_path = os.path.join(project_root, "data", "processed", "processed_data.csv")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data.to_csv(save_path, index=False)
    print(f"✅ 处理后的数据已保存到: {save_path}")

if __name__ == "__main__":
    # 测试数据加载
    data = load_startup_data()
    if data is not None:
        explore_data(data)