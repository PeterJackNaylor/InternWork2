# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:37:50 2015

@author: pubuntu
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from Reader import Reader
from sklearn.metrics import confusion_matrix
from Traj_creator import Traj_data
from Randomforest import RandomForest_Autotunner,plot_matrix,Measure,check_rotate

    
def MitoseClassif(obj_norm,
                  y_name_3state="Type",classif_Mitose="MitoseOrNot",
                  num_str="0015"):
    print "\n We first load the unnormalized data: \n"                  
    
    if os.path.isfile("H2b_data.csv"):
        print "The file existed so I loaded it."
        H2b = Traj_data(file_name="H2b_data.csv",pkl_traj_file="./Pkl_file") 
    
    else:    
        H2b=Traj_data() 
    
        H2b.extracting(num_str,"both_channels_0015.hdf5",'primary') 
        ## Extracting the hdf5 file for the primary channel (H2b)
    
        H2b.Add_traj(normalize=False)## ,num_traj=10) ## (you can reduce the number of traj)
        ## Adding Alice's work on tracking to have trajectories
    
        file_loc="0015_PCNA.xml"
    
        H2b.label_finder(file_loc) 
        ## Finding associated labels by minimizing distance by click and distance of cell
    
        H2b.renaming_and_merge() 
        ## renaming the labels to have G1=="1", S=="S", G2=="2" and M=="M" 
        #This procedure may take a long time.
        
        H2b.data.to_csv('H2b_data.csv',index=False,header=True) 
    print "\n We train a classifier for mitosis or not: \n"
    obj_unnorm=H2b
    train_file="MitoseClassif.arff"

    train_1=Reader()
    train_1.arrf_read(train_file)
    train_1.renaming_for_mitosis()
    
    train_1.data["label"].value_counts()
    
    kfold=3
    
    if train_1.Var_missing[0] in train_1.data.columns:
        train_1.missing_features_data()
        
    values=[100 + i*10 for i in range(15)]
    model_1=RandomForest_Autotunner(values)
    
    model_1.tunning(train_1.data[train_1.names],train_1.data["label"],kfold,plot=True,fit_new_model=True)
    plt.show()
    
    model_1.cm_normalized = model_1.cm.astype('float') / model_1.cm.sum(axis=1)[:, np.newaxis]
    
    plot_matrix(model_1.cm_normalized,title="Normalized confusion matrix",names=["M","O","S"])
    plt.show()
    
    ## To reduce computation and none useless things, we remove instances that do not belong to trajectories.

    obj_norm.data=obj_norm.data.ix[pd.notnull(obj_norm.data["traj"]),obj_norm.data.columns]
    obj_unnorm.data=obj_unnorm.data.ix[pd.notnull(obj_unnorm.data["traj"]),obj_unnorm.data.columns]
    
    obj_norm.update()
    obj_unnorm.update()
    ## Predicting model 1
    
    index_no_missing=obj_norm.data[obj_norm.names].dropna(axis=0, how='any').index
    obj_norm.data.ix[index_no_missing,classif_Mitose]=model_1.predict(obj_unnorm.data.ix[index_no_missing,train_1.names]) 
    ## Carefull, we put the unnormalized data in the above prediction.
    print "\n A bit of statistics on the overall predictions: \n"
    print "Frequency of predicted values for the Mitosis or not classifier: \n"
    print obj_norm.data[classif_Mitose].value_counts()
    
    
    print "\n We were however not able to predict %d instances because of missing values" % (obj_norm.data.shape[0]-len(index_no_missing))
    
    obj_norm.data
    
    obj_norm.update()
    
        ### Giving priority to the first classif...
    model_1.names_to_give=train_1.names
    return(obj_norm,model_1)

def EmissionMat(model_1,model_n_m):
    print "\n We compute the emission state probability matrix from the confusion matrix for the first classifier: \n"
    
    X3=model_n_m.cm_normalized
    X3=X3.T
    X3=np.array([X3[0],X3[2],X3[1]])
    X3=X3.T
    
    EmissionMat=np.zeros(shape=(5,5))
    EmissionMat[0,0]=model_1.cm_normalized[0,0]
    EmissionMat[4,4]=model_1.cm_normalized[0,0]
    EmissionMat[1:5,0]=(1-model_1.cm_normalized[0,0])/3
    EmissionMat[0:4,4]=(1-model_1.cm_normalized[0,0])/3
    
    ### Bricolage
    
    EmissionMat[1:4,1:4]=X3
    EmissionMat[1:4,1:3]+=-EmissionMat[3,0]*2/3
    
    ### On modifie car la diag n'est pas assez bonne...
    EmissionMat[3,2:4]=[0.4,0.5]
    
    EmissionMat[0,1:4]=sum(model_1.cm_normalized[0,1:3])/3
    EmissionMat[4,1:4]=sum(model_1.cm_normalized[0,1:3])/3
    
    EmissionMat=abs(EmissionMat).astype('float') / abs(EmissionMat).sum(axis=1)[:, np.newaxis]
    ## Put something better then abs... 
    
    plot_matrix(EmissionMat,title="Emission matrix",names=["M_B","G1","S","G2","M_E"])
    plt.show()
    return(EmissionMat)
    
import pdb
def prep_for_R(obj_norm,classif_3state="3state",classif_final="Pred_fusion",classif_Mitose="MitoseOrNot",num_str="0015"):
    def f(value_1,value_2):
        if value_1=="M":
            return(value_1)
        else:
            return(value_2)
    obj_norm.data[classif_final]=obj_norm.data.apply(lambda r: f(r[classif_Mitose],r[classif_3state]),axis=1)
    Mito=obj_norm.data[classif_final].dropna(axis=0, how='any')       
    print "\n We prioritize our predictor of mitosis events before the 3 state classfier giving \n us a four state classifier. \n"
    print "Frequency of predicted values for the 4 state classifier: \n"
    pdb.set_trace()
    print Mito.value_counts()
    obj_norm.update()
    
    ##First we are going to seperate beginning M's and ending M's
    if num_str+"_id_frame" in obj_norm.data.columns:
        obj_norm.data=obj_norm.data.sort_values(['traj', num_str+"_id_frame"], ascending=[1, 1])        
         ##First we are going to seperate beginning M's and ending M's
        for i in range(len(obj_norm.trajectories)):
            new_obs=np.array(obj_norm.data.ix[obj_norm.data["traj"]==i,classif_final])
            n_obs=len(new_obs)
            for j in range(n_obs/2):
                if new_obs[j]=='M':
                    new_obs[j]='B'  #Beginning
            obj_norm.data.ix[obj_norm.data["traj"]==i,classif_final]=new_obs
        obj_norm.data.ix[obj_norm.data[classif_final]=='M',classif_final]='E' #Ending
    else:
        obj_norm.data=obj_norm.data.sort_values(["Well",'traj',"Frame"], ascending=[1, 1, 1])
        subset=obj_norm.data[["Well","traj"]].drop_duplicates()
        tuples = [tuple(x) for x in subset.values]
        for i_well,i_traj in tuples:
            new_obs=np.array(obj_norm.data.ix[(obj_norm.data["Well"]==i_well) & (obj_norm.data["traj"]==i_traj),classif_final])
            n_obs=len(new_obs)
            for j in range(n_obs/2):
                if new_obs[j]=='M':
                    new_obs[j]='B'  #Beginning
            obj_norm.data.ix[(obj_norm.data["Well"]==i_well) & (obj_norm.data["traj"]==i_traj),classif_final]=new_obs
        obj_norm.data.ix[obj_norm.data[classif_final]=='M',classif_final]='E' #Ending

    data=obj_norm.data.ix[pd.notnull(obj_norm.data["traj"]),["traj",classif_final]]
    
    
    data.ix[data[classif_final]=='2',classif_final]="4"
    data.ix[data[classif_final]=='1',classif_final]="2"
    data.ix[data[classif_final]=='B',classif_final]="1"
    data.ix[data[classif_final]=='E',classif_final]="5"
    data.ix[data[classif_final]=='S',classif_final]="3"
    
    return(obj_norm,data)
def final_classif_HMM(data,obj_norm,
                  y_name_3state="Type",classif_Mitose="MitoseOrNot",
                  classif_3state="3state",classif_final="Pred_fusion",
                  ratio=5.9/60,obs_number=0):
    print "Here we are going to join the corrected data (from R) to our current data in Python \n "
    data.ix[data.HMM==1,"HMM"]="M"
    data.ix[data.HMM==2,"HMM"]="1"
    data.ix[data.HMM==3,"HMM"]="S"
    data.ix[data.HMM==4,"HMM"]="2"
    data.ix[data.HMM==5,"HMM"]="M"
    to_join=pd.Series(data["HMM"])
    to_join.index=[int(el) for el in to_join.index]
    obj_norm.data=obj_norm.data.join(to_join)
    obj_norm.update()
    
    print "Recap of our data: \n " 
    print obj_norm.train[["traj","Type",classif_Mitose,classif_3state,classif_final,"HMM"]].head()
    
    
    i=0
    G1=[]
    S=[]
    G2=[]
    CC=[]
    print "We are going to count the lengths of the G1 phase, the S phase and the G2 phase: \n"
    print "To quickly asses we print the trajectory and his corrected trajectory, for sequence number:" + str(obs_number)
    for el in obj_norm.Group_of_traj:
        new_obs=el[1]["HMM"]
        if i==obs_number:
            test=np.array(el[1][classif_final])
            test_hmm=np.array(el[1]["HMM"])
            print classif_final+": \n"
            print test
            print "\n Corrected HMM: \n"
            print test_hmm
        i+=1
        if not check_rotate(new_obs):
            G1.append(Measure(new_obs,'1',_last=True))
            S.append(Measure(new_obs,'S',_last=True,_first=True))
            G2.append(Measure(new_obs,'2',_first=True))
            CC.append(Measure(new_obs,'M'))
        elif not check_rotate(new_obs[:-1]):
            G1.append(Measure(new_obs[:-1],'1',_last=True))
            S.append(Measure(new_obs[:-1],'S',_last=True,_first=True))
            G2.append(Measure(new_obs[:-1],'2',_first=True))
            CC.append(Measure(new_obs[:-1],'M'))
        elif not check_rotate(new_obs[:-2]):
            G1.append(Measure(new_obs[:-2],'1',_last=True))
            S.append(Measure(new_obs[:-2],'S',_last=True,_first=True))
            G2.append(Measure(new_obs[:-2],'2',_first=True))
            CC.append(Measure(new_obs[:-2],'M'))
        else:
            G1.append(-1)
            S.append(-1)
            G2.append(-1)
            CC.append(-1)
            
    ratio=5.9/60
    G1_p=[el*ratio for el in G1 if el>-1]
    S_p= [el*ratio for el in S  if el>-1]
    G2_p=[el*ratio for el in G2 if el>-1]
    CC_p=[el*ratio for el in CC if el>-1] 
    res = {'mean' : pd.Series([np.mean(G1_p), np.mean(S_p), np.mean(G2_p),np.mean(CC_p)], index=['G1', 'S', 'G2','CellCycle']),
           'Standard deviation' : pd.Series([np.std(G1_p),np.std(S_p),np.std(G2_p),np.std(CC_p)], index=['G1', 'S', 'G2','CellCycle']),
           'Accepted trajectories': pd.Series([len(G1_p),len(S_p),len(G2_p),len(CC_p)], index=['G1', 'S', 'G2','CellCycle'])
              }
              
    
    temp_X=obj_norm.train.ix[pd.notnull(obj_norm.train["HMM"]),["HMM","Type"]]
    print temp_X["Type"].value_counts()
    cm=confusion_matrix(temp_X.Type,temp_X.HMM)
    
    print "We reach an accuracy of %5.3f \n" %(float(cm.trace())/cm.sum())
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plot_matrix(cm_normalized,title="Confusion matrix for the HMM classification")

    return(obj_norm,pd.DataFrame(res))