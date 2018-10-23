#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.metrics import accuracy_score
"""

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=10,n_jobs=-1)

clf.fit(features_train,labels_train)
pred1 = clf.predict(features_test)


print ("knn",accuracy_score(labels_test,pred1))

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train,labels_train)
prednb = clf.predict(features_test)
print ("nb",accuracy_score(labels_test,prednb))

from sklearn.svm import SVC
clf = SVC(kernel="rbf",C=10000.0)
clf.fit(features_train,labels_train)
predsvm = clf.predict(features_test)
print ("svc",accuracy_score(labels_test,predsvm))

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=4)
clf.fit(features_train,labels_train)
preddtc = clf.predict(features_test)
print ("dtc",accuracy_score(labels_test,preddtc))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(random_state = 11,max_depth = 200,min_samples_split=3)
clf1 = AdaBoostClassifier(base_estimator = DTC,n_estimators=500,random_state=7,learning_rate=0.4)
clf1.fit(features_train,labels_train)
pred = clf1.predict(features_test)

print ("abc",accuracy_score(labels_test,pred))
"""
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=19,max_depth=200,min_samples_split=3)
clf.fit(features_train,labels_train)
predrf = clf.predict(features_test)

print ("rf",accuracy_score(labels_test,predrf))

"""
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
"""
