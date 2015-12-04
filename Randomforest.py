# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:00:54 2015

@author: naylor
"""
import pandas as pd
from sklearn import grid_search
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np
import pylab
import time
import matplotlib.pyplot as plt

class RandomForest_Autotunner(RandomForestClassifier):
    def __init__(self, n_range,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 n_estimators=10):
        super(RandomForest_Autotunner, self).__init__(  bootstrap=bootstrap, class_weight=class_weight, criterion=criterion,
                                                        max_depth=max_depth, max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                                        min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                                        min_weight_fraction_leaf=min_weight_fraction_leaf, n_estimators=n_range[0], n_jobs=n_jobs,
                                                        oob_score=oob_score, random_state=random_state, verbose=verbose,
                                                        warm_start=warm_start)
        self.n_range=n_range
    def tunning(self,X,y,kfold=3,plot=True,fit_new_model=True,opti=False):
        ##optimize with grid search
    
        tic = time.clock()
        if not opti:
            n,p=X.shape
            MSE=np.array([0.0]*len(self.n_range))
            true_index=np.array(X.index)
            skf = StratifiedKFold(y, n_folds=kfold,shuffle=True)
            j=0
            Confusions_matrix=[]
            for tree_n in self.n_range:
                y_pred = pd.Series(np.array(["N"]*n), index=X.index)
                for train_index,test_index in skf:
                    train_index=true_index[train_index]
                    test_index=true_index[test_index]
                    X_train=X.ix[train_index,]
                    X_test=X.ix[test_index,]
                    y_train=[y[i] for i in train_index]
                    rf=RandomForestClassifier( n_estimators = tree_n,
                                                  max_depth=self.max_depth, 
                                                  min_samples_split=self.min_samples_split, 
                                                  min_samples_leaf=self.min_samples_leaf, 
                                                  min_weight_fraction_leaf=self.min_weight_fraction_leaf, 
                                                  max_features=self.max_features, 
                                                  max_leaf_nodes=self.max_leaf_nodes,
                                                  bootstrap=self.bootstrap,
                                                  oob_score=self.oob_score,
                                                  n_jobs=self.n_jobs,
                                                  random_state=self.random_state,
                                                  verbose=self.verbose,
                                                  warm_start=self.warm_start,
                                                  class_weight=self.class_weight)
                    rf=rf.fit(X_train,y_train)
                    y_pred[test_index]=rf.predict(X_test)
                cm=confusion_matrix(y,y_pred)
                Confusions_matrix.append(cm)
                MSE[j]=float(cm.trace())/float(n)
                j+=1
            i_est=np.argmax(MSE)
            if fit_new_model:
                self.n_estimators=self.n_range[i_est]
                self.fit(X,y)
                self.cm=Confusions_matrix[i_est]
            self.cv_score=max(MSE)
            self.MSE=MSE
            if plot:
                pylab.plot(self.n_range,MSE)
        else:
            parameters = {'n_estimators': self.n_range}
            self.clf = grid_search.GridSearchCV(self, parameters,n_jobs=self.n_jobs, cv = kfold)
            self.clf.fit(X, y)
        toc = time.clock()
        print "Processing time: %f in sec" % (toc - tic)



def plot_matrix(cm, title='Confusion matrix',names=['G1','G2','M','S'], cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


##    First and last resp. forces the length to be valid only if the trajectory doesn't start 
##    or finishes by the given state

   
def check_rotate(new_obs):
    new_obs=np.array(new_obs)
    loop=False
    in_loop=False
    Bad=False
    for i in range(len(new_obs)):
        if ((new_obs[i]=='1') and (not in_loop)):
            if loop:
                Bad=True
            else:
                in_loop=True            
        elif new_obs[i]!='1' and in_loop:
            in_loop=False
            loop=True
    return(Bad)

def Measure(new_obs,state,_last=False,_first=False):
    new_obs=np.array(new_obs)
    if state!="M": ##For cellcycle
        if state not in new_obs:
            return(-1)
        else:
            _length=[el for el in new_obs if el==state]
            if len(_length)>0:
                if (_last and _first):
                    if ((new_obs[0]!=state) and (new_obs[-1]!=state)):
                        return(len(_length))
                    else:
                        return(-1)
                elif _last:
                    if new_obs[-1]!=state:
                        return(len(_length))
                    else:
                        return(-1)
                elif _first:
                    if new_obs[0]!=state:
                        return(len(_length))
                    else:
                        return(-1)
                else:
                    return(len(_length))
    else:
        if (('2' in new_obs) and ('1' in new_obs) and ('S' in new_obs)):
            return(len([el for el in new_obs if el!=state]))
        else:
            return(-1)

## print Measure(test,'1',_last=True),Measure(test,'S',_last=True,_first=True),Measure(test,'2',_first=True),Measure(test,'M')
"""        
_range=[100+i*10 for i in range(10)]
print _range

trys=RandomForest_Autotunner(_range,oob_score=True)
trys.tunning(train.data[train.names],train.data['label'],3)
"""


import os
from Traj_creator import Traj_data
num_str="0015"

if os.path.isfile("H2B_N_F_A.csv"):
    print "The file existed so I loaded it."
    H2B_N_F_A = Traj_data(file_name="H2B_N_F_A.csv")#,pkl_traj_file="/home/pubuntu/Documents/InternWork2/Pkl_file") 
    H2B_N_F_A.caract="Normalized by dividing by average"
else:    
    H2B_N_F_A=Traj_data()#(pkl_traj_file="/home/pubuntu/Documents/InternWork2/Pkl_file") 

    H2B_N_F_A.extracting(num_str,"both_channels_0015.hdf5",'primary') 
    ## Extracting the hdf5 file for the primary channel (H2b)
    
#    H2B_N_F_A.add_error() ## We had it so that the data won't have to do 0/0

    H2B_N_F_A.Add_traj(normalize=True,all_traj=True,average=True,diff=False)## ,num_traj=10) ## (you can reduce the number of traj)
    ## Adding Alice's work on tracking to have trajectories

    file_loc="0015_PCNA.xml"

    H2B_N_F_A.label_finder(file_loc) 
    ## Finding associated labels by minimizing distance by click and distance of cell

    H2B_N_F_A.renaming_and_merge() 
    ## renaming the labels to have G1=="1", S=="S", G2=="2" and M=="M" 
    #This procedure may take a long time.

    H2B_N_F_A.data.to_csv('H2B_N_F_A.csv',index=False,header=True)
    
    
list_obj=[ H2B_N_F_A]
kfold=3
D={}
instances_to_keep=H2B_N_F_A.train[pd.notnull(H2B_N_F_A.train.traj)].index
for obj in list_obj:
    if obj.Var_missing[0] in obj.train.columns:
        obj.missing_features_train()
    if obj.Var_missing[0] in obj.data.columns:
        obj.missing_features_data()
        
    #instances_to_keep=pd.notnull(obj.train.traj)

    values=[100 + i*10 for i in range(30)]

    model=RandomForest_Autotunner(values,n_jobs=1)

    model.tunning(obj.train.ix[instances_to_keep,obj.names],obj.train.ix[instances_to_keep,"Type"],
                  kfold,plot=False,fit_new_model=True,opti=True) #fit new model to get cm
    plt.show()

    i_=np.argmax(model.MSE)
    n_tree=values[i_]
    model.cm_normalized = model.cm.astype('float') / model.cm.sum(axis=1)[:, np.newaxis]
    D[obj.caract]={"tree_tunning":n_tree,
                   "best accuracy":max(model.MSE),
                   "Accuracy vector":model.MSE,
                   "Confusion matrix":model.cm,
                   "Normalized confusion matrix":model.cm_normalized,
                   "Training sample":str(obj.train.ix[instances_to_keep,obj.names].shape[0])
                  }