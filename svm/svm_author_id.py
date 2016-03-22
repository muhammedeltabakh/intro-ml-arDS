#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("Desktop/my_work/ud120-projects/tools")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np 



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

x1 = features_train#[:len(features_train)/100]
x2 = features_test

y1 = labels_train#[:len(labels_train)/100]
y2 = labels_test

clf = SVC(gamma = .001, C = 10000, kernel = 'rbf')

pred = clf.fit(x1,y1).predict(x2)

print len(pred)

n = []
for e in pred:
	if e == 1:
		n.append(e)
print len(n)

#print pred[10]
#print pred[26]
#print pred[50]
#accuracy = accuracy_score(pred, y2)

#print accuracy 
#########################################################


