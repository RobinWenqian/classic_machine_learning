from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

digits=load_digits()
digits.data.shape
X_train, X_test, y_train, y_test=train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
y_train.shape
y_test.shape