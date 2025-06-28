from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas as pd
from scipy.stats import uniform, randint

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

x_train,x_test,y_train,y_test = train_test_split(features_scaled, labels, test_size=0.1, random_state=10)

classifier = DecisionTreeClassifier()


# 定义参数网格范围
param_dist = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 随机化搜索与交叉验证
random_search = RandomizedSearchCV(classifier, param_distributions=param_dist, n_iter=10, cv=5,
                                   random_state=42, n_jobs=-1)
random_search.fit(x_train, y_train)

# 查看结果
print("最佳参数组合:", random_search.best_params_)
print("最佳交叉验证分数:", random_search.best_score_)

# 评估模型
best_dt_classifier = random_search.best_estimator_
accuracy = best_dt_classifier.score(x_test, y_test)
print("测试集准确率:", accuracy)