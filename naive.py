import pandas as pd
import numpy as np
from sklearn import *
from sklearn.naive_bayes import *
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/nagad/Downloads/covid.csv")

le=preprocessing.LabelEncoder()

x=data[['pc','wbc','ast','bc','ldh']].apply(le.fit_transform).values
y = le.fit_transform(data['diagnosis'].values)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

model=MultinomialNB()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

print("Accuracy:\n", accuracy_score(ytest, ypred))
print("Classification Report:\n", classification_report(ytest, ypred))

prob=model.predict_proba(xtest)[:,1]

fpr,tpr,_=roc_curve(ytest,prob)

plt.plot(lr_fpr,lr_tpr,marker='.',label='Naive Bayes Classifier')

plt.xlabel('false positive rate')
plt.ylabel('true positive rate')

plt.legend()
plt.show()
