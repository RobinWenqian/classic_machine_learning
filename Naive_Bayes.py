from sklearn.datasets import fetch_20newsgroups
#即时通过新闻数据抓取器抓取新闻
news = fetch_20newsgroups(subset='all')

#查验数据规模和细节
print (len(news.data))
print(news.data[0])

#对数据进一步处理随机采样一部分用于测试
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(news.data, news.target, test_size=0.25, random_state=33)

#文本特征向量转化
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

#导入贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_predict = mnb.predict(X_test)

#评估分类性能
from sklearn.metrics import classification_report
print('The accuracy of Naive Bayes is',mnb.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=news.target_names))