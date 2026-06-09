#!/usr/bin/env python3
"""测试 Prophet 模型修复"""
import pandas as pd
from hp_ml.models_extended import ProphetWrapper

# 创建测试数据
test_data = pd.DataFrame({
    'code': ['159300', '159300', '159300', '510300', '510300', '510300'],
    'date': pd.date_range('2024-01-01', periods=6, freq='D'),
    'feature_1': [1, 2, 3, 4, 5, 6],
    'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
})

y = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01, 0.02])

print("测试数据：")
print(test_data)
print("\n目标值：")
print(y)

try:
    # 测试 Prophet 训练
    print("\n开始训练 Prophet 模型...")
    model = ProphetWrapper()

    X_train = test_data[['code', 'date', 'feature_1', 'feature_2']]
    model.fit(X_train, y)

    print("✅ Prophet 模型训练成功！")

    # 测试预测
    print("\n开始预测...")
    predictions = model.predict(X_train)
    print(f"预测结果：{predictions}")
    print("✅ Prophet 模型预测成功！")

except Exception as e:
    print(f"❌ 错误：{e}")
    import traceback
    traceback.print_exc()
