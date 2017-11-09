import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu//wiki/pub/Main/DataSets/titanic.txt')

#选取pclass,age,sex作为特征判断乘客是否能生还
X = titanic[['pclass','age','sex']]
y=titanic['survived']

#将缺失的年龄信息用全体的平均年龄代替
X['age'].fillna(X['age'].mean(), inplace=True)

#分割25%的乘客数据用于测试集
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=33)

#对类别型特征进行转化，成为特征向量
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

#使用随机森林做训练与预测分析
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred=rfc.predict(X_test)

#使用梯度提升决策树
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict((X_test))

#评估两种分类模型
from sklearn.metrics import classification_report
print('The accuracy of random forest is',rfc.score(X_test, y_test))
print(classification_report(rfc_y_pred, y_test))

print('The accuracy of gradient tree is',gbc.score(X_test, y_test))
print(classification_report(gbc_y_pred, y_test))