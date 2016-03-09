#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
import numpy as np 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score   
import sys
from time import time
sys.path.append("Desktop/my_work/ud120-projects/tools")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################

x1 = features_train
x2 = features_test

y1 = labels_train
y2 = labels_test

clf = GaussianNB()

pred = clf.fit(x1,y1).predict(x2)

accuracy = accuracy_score(pred, y2)
print accuracy
#########################################################


