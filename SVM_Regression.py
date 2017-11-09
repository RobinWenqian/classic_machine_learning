#从sklearn导入波士顿房价数据读取器
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.DESCR)

#分割数据
from sklearn.cross_validation import train_test_split
import numpy as np
X = boston.data
y = boston.target
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=33)

#数据标准化处理
from sklearn.preprocessing import StandardScaler
#初始化标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()
#分别对训练和测试数据标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

#导入SVM用于回归分析
from sklearn.svm import SVR
#使用线性核函数配置SVM训练预测
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, y_train)
linear_svr_y_predict = linear_svr.predict(X_test)
#使用多项式核函数配置SVM训练
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = linear_svr.predict(X_test)
#使用径向基核函数配置
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

#使用R_square，MSE，MAE三个指标对三种SVM配置评估
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print('R-squared value of linear SVR is', linear_svr.score(X_test, y_test))
print('The mean squared error of linear SVR is', mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))
print('The mean absolute error of linear SVR is', mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))
