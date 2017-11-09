from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

# 加载数据
iris = load_iris()
data_X = iris.data
data_y = iris.target

# 数据维度、特征与目标值的前3项
print('data:', data_X.shape, data_y.shape)
print('features:', data_X[:3, :])
print('target:', data_y[:3])

# 数据切分
train_X, test_X, train_y, test_y = train_test_split(data_X,
data_y, test_size=0.2)

# 训练数据与测试数据的维度
print('train:', train_X.shape, train_y.shape)
print('test: ', test_X.shape, test_y.shape)

from sklearn.neighbors import KNeighborsClassifier

# 构建knn模型
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

# 拟合数据
knn.fit(train_X, train_y)

# 预测
preds = knn.predict(test_X)

print('knn model:', knn)
print('First 3 pred:',preds[:3])

from pprint import pprint

# 使用测试的特征与测试的目标值
print(knn.score(test_X, test_y))

from sklearn.metrics import precision_recall_fscore_support

# 打印出三个指标
scores = precision_recall_fscore_support(test_y, preds)
pprint(scores)

from sklearn.externals import joblib

# 保存模型
joblib.dump(knn, 'knn_model.pkl')
