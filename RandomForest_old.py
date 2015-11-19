# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 12:45:41 2015

@author: Peter-Jack
"""

import pandas as pd
import numpy as np


from auxilary_func import *

norm="_n"
regression="rf"
phase=3


a=40
b=45
pr=10

a_c=1
b_c=20
pr_c=0.1

kfold=5
gamma=0.01
C=51
n_tree=400

n_range=np.array(range(a,b))*pr
c_range=np.array(range(a_c,b_c))*pr_c

val_=[]
Values=[]

data=pd.read_csv('data'+norm+'_M.csv',sep=";",encoding='utf-8')
data.index=data["id"]

names=[str(el) for el in range(235) if np.var(data[str(el)])!=0]

y=data["y"]
y=pd.DataFrame([bij(el,phase) for el in y],index=y.index)
y.columns=["y"]

if phase!=4:
    data=data[data.y!=3]

X=data[names]

y=y[y["y"]!="-1"]
y.shape
X.shape
###Scalling the data
names=[el for el in X.columns if np.std(X[el])!=0]
X=X[names]
##for k in names:
##    mean_k=np.mean(X[str(k)])
##    std_k=np.std(X[str(k)])
##    X[str(k)]=(X[str(k)]-mean_k)/std_k

MSE=np.array([0.0]*len(n_range))
i=0
for n_tree in n_range:
   y_true,y_pred=Classif(regression,X,y,n_tree,gamma,C,kfold)
   cm=confusion_matrix(y_true,y_pred)
   t=[j for j in range(np.sum(cm)) if y_true[j]==y_pred[j]]
   MSE[i]=1-float(cm.trace())/cm.sum()
   i=i+1

print min(MSE)

import pylab
pylab.plot(n_range,MSE)
