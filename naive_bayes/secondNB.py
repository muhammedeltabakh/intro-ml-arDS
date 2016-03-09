from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

x = iris.data 
y = iris.target

x1 = x[0:30]
x2 = x[30:150]

y1 = y[0:30]
y2 = y[30:150]

clf = GaussianNB()

pred = clf.fit(x1,y1).predict(x2)

accuracy = accuracy_score(pred, y2)

print (accuracy)
