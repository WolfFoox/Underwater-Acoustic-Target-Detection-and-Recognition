import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

n_classes = 3  # 多分类任务
file_name = "F:/document/海天探索者/水下信标/波形/valid/dataALE.xlsx"
data=pd.read_excel(io=file_name)

features=data.iloc[:,1:-1]
labels=data.iloc[:,-1]



print(type(labels))
print(labels.unique())
print(len(labels.unique()))

# 数据预处理
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)  # 特征缩放

X_train,X_test,y_train,y_test = train_test_split(features_scaled, labels, test_size=0.1, random_state=10)

# 转换为DMatrix对象
train_dmatrix = xgb.DMatrix(X_train, label=y_train)
test_dmatrix = xgb.DMatrix(X_test, label=y_test)

# 设置XGBoost参数
params = {
    'objective': 'multi:softmax',
    'num_class': n_classes,
    'max_depth': 6,
    'learning_rate': 0.1,
    'eval_metric': 'mlogloss'
}

# 训练模型
model = xgb.train(params, train_dmatrix, num_boost_round=500, evals=[(test_dmatrix, 'Test')], early_stopping_rounds=50)

# 预测
y_pred = model.predict(test_dmatrix)

# 评估模型
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
