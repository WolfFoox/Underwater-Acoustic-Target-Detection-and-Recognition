from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
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
    'loss': ['log', 'hinge'],
    'penalty': ['l1', 'l2'],
    'alpha': uniform(0.0001, 0.01),
    'learning_rate': ['constant', 'invscaling'],
    'eta0': uniform(0.01, 1),
    'max_iter': randint(100, 300)
}
# 创建 SGDClassifier 模型
sgd_clf = SGDClassifier()
# 使用随机搜索进行参数调节
random_search = RandomizedSearchCV(sgd_clf, param_distributions=param_dist, n_iter=10, cv=5)
random_search.fit(x_train, y_train)

# 输出最佳参数组合
print("最佳参数组合：", random_search.best_params_)

# 在测试集上进行预测
y_pred = random_search.predict(x_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)