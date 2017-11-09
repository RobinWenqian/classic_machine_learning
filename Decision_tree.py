import pandas as pd
#收集网络泰坦尼克乘客数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu//wiki/pub/Main/DataSets/titanic.txt')
titanic.head()

#查看数据的统计特性
titanic.info()

#特征选择
X=titanic[['pclass','age','sex']]
y=titanic['survived']

#查看当前选择特征
X.info()

#由于age里面的数据维数缺失，用平均数或者中位数填补空缺
X['age'].fillna(X['age'].mean(),inplace=True)
X.info()

#数据分割
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=33)

#使用scikit-learn.feature_extraction中的特征转换器（将类别特征转化为bool）
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)

#转换特征后，凡是类别特征都单独抽出来列成一列
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
print(vec.feature_names_)

#也对测试数据的特征进行转换
X_test = vec.transform(X_test.to_dict(orient='record'))

#从sklearn.tree导入决策树分类器
from sklearn.tree import DecisionTreeClassifier

#使用默认配置初始化决策树
dtc= DecisionTreeClassifier()

#使用分割过的训练数据进行模型学习
dtc.fit(X_train,y_train)
y_predict=dtc.predict(X_test)

#用决策树预测乘客生还性能
from sklearn.metrics import classification_report
print(dtc.score(X_test, y_test))
print(classification_report(y_predict, y_test, target_names=['died','survived']))

#保存模型
from sklearn.externals import joblib
joblib.dump(dtc, 'Desion_tree.pkl')
