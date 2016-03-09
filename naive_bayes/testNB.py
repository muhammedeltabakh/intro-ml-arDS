import numpy as np
from random import shuffle
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

x = iris.data
y = iris.target

xn = []

yn = []

idx = [i for i in range(150)]
shuffle (idx)

for i in idx:
	xn.append(x[i])
	yn.append(y[i])

y1 = yn[30:70]
y2 = yn[80:150]

x1 = xn[30:70]
x2 = xn[80:150]

clf = GaussianNB()

clf = clf.fit(x2,y2)
pred = clf.predict(x1)

accuracy = accuracy_score(pred,y1)

print accuracy