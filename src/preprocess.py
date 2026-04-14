import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def handle_missing_values(data):
    """处理缺失值"""
    data_clean = data.copy()
    
    print("🔧 处理缺失值...")
    
    # 检查缺失值
    missing_counts = data_clean.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"发现缺失值: {missing_counts[missing_counts > 0].to_dict()}")
        
        # 数值列用中位数填充
        numeric_cols = ['Funding Rounds', 'Funding Amount (M USD)', 'Valuation (M USD)', 
                       'Revenue (M USD)', 'Employees', 'Market Share (%)']
        for col in numeric_cols:
            if col in data_clean.columns and data_clean[col].isnull().sum() > 0:
                median_val = data_clean[col].median()
                data_clean[col].fillna(median_val, inplace=True)
                print(f"  {col}: 用中位数 {median_val:.2f} 填充")
        
        # 分类列用众数填充
        categorical_cols = ['Industry']
        for col in categorical_cols:
            if col in data_clean.columns and data_clean[col].isnull().sum() > 0:
                mode_val = data_clean[col].mode()[0]
                data_clean[col].fillna(mode_val, inplace=True)
                print(f"  {col}: 用众数 '{mode_val}' 填充")
    else:
        print("✅ 无缺失值")
    
    return data_clean

def encode_categorical_features(data):
    """编码分类特征"""
    data_encoded = data.copy()
    
    print("\n🔤 编码分类特征...")
    
    # 行业编码（使用频率编码，避免one-hot维度爆炸）
    industry_freq = data_encoded['Industry'].value_counts()
    data_encoded['Industry_Frequency'] = data_encoded['Industry'].map(industry_freq)
    
    # 行业热度编码（按频率排序）
    industry_rank = {industry: rank for rank, industry in enumerate(industry_freq.index, 1)}
    data_encoded['Industry_Rank'] = data_encoded['Industry'].map(industry_rank)
    
    print(f"行业数量: {len(industry_freq)}")
    print(f"行业分布: {industry_freq.to_dict()}")
    
    return data_encoded

def create_business_features(data):
    """创建业务特征"""
    data_features = data.copy()
    
    print("\n💡 创建业务特征...")
    
    # 1. 融资效率特征
    data_features['Funding_Efficiency'] = data_features['Revenue (M USD)'] / data_features['Funding Amount (M USD)']
    data_features['Funding_Efficiency'] = data_features['Funding_Efficiency'].replace([np.inf, -np.inf], 0)
    
    # 2. 估值收入比（P/S比率）
    data_features['PS_Ratio'] = data_features['Valuation (M USD)'] / data_features['Revenue (M USD)']
    data_features['PS_Ratio'] = data_features['PS_Ratio'].replace([np.inf, -np.inf], 0)
    
    # 3. 人均收入
    data_features['Revenue_Per_Employee'] = data_features['Revenue (M USD)'] / data_features['Employees']
    data_features['Revenue_Per_Employee'] = data_features['Revenue_Per_Employee'].replace([np.inf, -np.inf], 0)
    
    # 4. 融资轮次效率
    data_features['Funding_Per_Round'] = data_features['Funding Amount (M USD)'] / data_features['Funding Rounds']
    data_features['Funding_Per_Round'] = data_features['Funding_Per_Round'].replace([np.inf, -np.inf], 0)
    
    # 5. 市场规模指数
    data_features['Market_Potential'] = data_features['Market Share (%)'] * data_features['Revenue (M USD)']
    
    # 6. 资本效率指数
    data_features['Capital_Efficiency'] = (
        data_features['Revenue (M USD)'] * data_features['Market Share (%)']
    ) / data_features['Funding Amount (M USD)']
    data_features['Capital_Efficiency'] = data_features['Capital_Efficiency'].replace([np.inf, -np.inf], 0)
    
    print("✅ 业务特征创建完成")
    
    return data_features

def remove_outliers(data, method='iqr', threshold=3):
    """移除异常值"""
    data_clean = data.copy()
    
    print("\n📊 处理异常值...")
    
    numeric_cols = ['Funding Amount (M USD)', 'Valuation (M USD)', 'Revenue (M USD)', 
                   'Employees', 'Market Share (%)']
    
    outliers_count = 0
    
    if method == 'iqr':
        # IQR方法
        for col in numeric_cols:
            if col in data_clean.columns:
                Q1 = data_clean[col].quantile(0.25)
                Q3 = data_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data_clean[(data_clean[col] < lower_bound) | (data_clean[col] > upper_bound)]
                outliers_count += len(outliers)
                
                # 用边界值替换异常值
                data_clean[col] = np.where(data_clean[col] < lower_bound, lower_bound, data_clean[col])
                data_clean[col] = np.where(data_clean[col] > upper_bound, upper_bound, data_clean[col])
    
    print(f"处理了 {outliers_count} 个异常值")
    
    return data_clean

def prepare_features_and_target(data):
    """准备特征和目标变量"""
    print("\n🎯 准备特征和目标变量...")
    
    # 选择特征列
    base_features = [
        'Funding Rounds', 'Funding Amount (M USD)', 'Valuation (M USD)',
        'Revenue (M USD)', 'Employees', 'Market Share (%)',
        'Industry_Frequency', 'Industry_Rank'
    ]
    
    business_features = [
        'Funding_Efficiency', 'PS_Ratio', 'Revenue_Per_Employee',
        'Funding_Per_Round', 'Market_Potential', 'Capital_Efficiency'
    ]
    
    all_features = base_features + business_features
    
    # 只保留存在的特征
    available_features = [col for col in all_features if col in data.columns]
    
    X = data[available_features]
    y = data['Profitable']
    
    print(f"特征数量: {len(available_features)}")
    print(f"特征列表: {available_features}")
    print(f"目标变量分布: {y.value_counts().to_dict()}")
    
    return X, y, available_features

def split_data(X, y, test_size=0.2, random_state=42):
    """分割训练集和测试集"""
    print(f"\n📊 数据集分割 (测试集比例: {test_size})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"训练集目标分布: {y_train.value_counts().to_dict()}")
    print(f"测试集目标分布: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """特征标准化"""
    print("\n⚖️ 特征标准化...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ 特征标准化完成")
    
    return X_train_scaled, X_test_scaled, scaler

def preprocess_data(data_path=None):
    """完整的数据预处理流程"""
    print("🚀 开始数据预处理流程")
    print("=" * 50)
    
    # 1. 加载数据
    from .data_loader import load_startup_data
    data = load_startup_data(data_path)
    
    if data is None:
        return None, None, None, None, None, None
    
    # 2. 处理缺失值
    data = handle_missing_values(data)
    
    # 3. 编码分类特征
    data = encode_categorical_features(data)
    
    # 4. 创建业务特征
    data = create_business_features(data)
    
    # 5. 处理异常值
    data = remove_outliers(data)
    
    # 6. 准备特征和目标
    X, y, feature_names = prepare_features_and_target(data)
    
    # 7. 分割数据集
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 8. 特征标准化
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("\n✅ 数据预处理完成!")
    print("=" * 50)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler

if __name__ == "__main__":
    # 测试预处理流程
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data()
    if X_train is not None:
        print(f"\n预处理结果:")
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")
        print(f"特征名称: {feature_names}")