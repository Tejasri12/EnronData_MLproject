#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL")
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

Y,x = targetFeatureSplit(data)
print len(data)
### your code below
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
trainx,testx,trainy,testy = train_test_split(x,Y,test_size=0.1, random_state=18)
reg = LinearRegression()
reg.fit(trainx,trainy)
pred = reg.predict(testx)



for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary,bonus)

plt.plot(testx,pred,color='blue')
plt.show()
print max(Y),max(x)

for k,v in data_dict.iteritems():
    if v["bonus"]!="NaN" and v["salary"]!="NaN":
        if v["bonus"] > 5000000 and v["salary"] > 1000000:
            print k,v
        
        
        


