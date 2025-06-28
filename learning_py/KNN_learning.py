from sklearn.neighbors import KNeighborsClassifier
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

# 定义参数分布
param_dist = {
    'n_neighbors': randint(low=1, high=10),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': randint(low=10, high=50)
}

# 创建KNN模型
knn = KNeighborsClassifier()

# 使用RandomizedSearchCV进行随机搜索
random_search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=10, cv=5)
random_search.fit(x_train, y_train)

# 输出最佳参数组合
print("最佳参数组合：", random_search.best_params_)

# 在测试集上评估模型性能
accuracy = random_search.score(x_test, y_test)
print("测试集准确率：", accuracy)