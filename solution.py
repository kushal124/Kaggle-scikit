import numpy as np
import pylab as pl
from sklearn import cross_validation
from sklearn import svm
from sklearn import linear_model

'''
Reading data 
'''

train_data = np.genfromtxt("train.csv" , delimiter=",")
train_labels = np.genfromtxt("trainLabels.csv", delimiter=",")
test_data = np.genfromtxt("test.csv", delimiter=",")

'''
Cross Validation data split
'''

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(train_data, train_labels, test_size = 0.4, random_state =1)

'''
Algorithms to be used :

'''
#linear model

clf_lin = linear_model.LinearRegression()
clf_lin.fit(X_train,Y_train)
score_lin = clf_lin.score(X_train,Y_train)

# Ridge model
clf_rid = linear_model.RidgeCV(alphas = [0.1,1.0,10.0])
clf_rid.fit(X_train,Y_train)
clf_rid.score(X_test,Y_test)	

## Prediction ##
prediction =

## Save Data ##

idcol = np.arange(start=1,stop=9001)
result = np.column_stack((idcol,prediction))

np.savetxt('prediction.csv',result,fmt='%d',delimiter=",")
