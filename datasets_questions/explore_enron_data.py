#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
cnt=0
"""
for k,v in enron_data.items():
    if v['poi']:
        cnt=cnt+1
print cnt
"""
print(len(enron_data))
#print enron_data["Skilling Jeffrey K".upper()]['exercised_stock_options']

"""
for x in ["Lay Kenneth L", "Skilling Jeffrey K", "Fastow Andrew S"]:
    print(enron_data[x.upper()]["total_payments"])
    #print(v["total_payments"])
    
#print enron_data["Prentice James"]
"""
import math
cnt1=0
tot=0
c=0
for k,v in enron_data.items():
    tot=tot+1
    if v['poi']:
        c=c+1
        print(v["poi"])
    if not v['salary'] == "NaN":
        cnt=cnt+1
    if v['total_payments'] == "NaN":
        cnt1=cnt1+1
        

print (c,cnt,cnt1,tot)
print(cnt1*100.0/tot)
