import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

data=pd.read_csv("C:/Users/nagad/Downloads/zoo_data.csv")

x=data.drop("1.7",axis=1)
y=data['1.7']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=42)

clf=DecisionTreeClassifier(criterion='entropy')

clf.fit(xtrain,ytrain)

ypred= clf.predict(xtest)

print("accuracy is:",accuracy_score(ytest,ypred))
print("rep:",classification_report(ytest,ypred))
print("Confusion Matrix is:",confusion_matrix(ytest,ypred))

fig=plt.figure(figsize=(25,20))
_=tree.plot_tree(clf)
