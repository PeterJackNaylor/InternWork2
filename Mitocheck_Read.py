# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:47:39 2015

@author: naylor
"""

import os 



# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:21:13 2015

@author: Peter-Jack
"""

import pandas as pd
import cPickle as pkl
import h5py
import numpy as np
from aux_FilterTraj import *



def File(Well,file_pkl,file_hdf5):

    fp=open(file_pkl,'r')

    a=pkl.load(fp)
    fp.close()
        
    first=a[a.keys()[0]].keys()[0]
    second=a[a.keys()[0]][first].keys()[0]
    
    Connexions=a['connexions between tracklets'][first][second]
    traj=a['tracklets dictionary'][first][second]
    trajectories=[traj.lstTraj[ind] for ind in range(len(traj.lstTraj))]
    
    movie_length=a['movie_length'][first][second]
    
    a=h5py.File(file_hdf5, "r+")  
    path_features="/sample/0/plate/"+first+"/experiment/"+second[0:(len(second)-3)]+"/position/1"+"/feature/primary__primary/object_features"
    path_id="/sample/0/plate/"+first+"/experiment/"+second[0:(len(second)-3)]+"/position/1/object/primary__primary"
    path_classification="/sample/0/plate/"+first+"/experiment/"+second[0:(len(second)-3)]+"/position/1"+"/feature/primary__primary/object_classification/prediction"
    
    path_features_names="definition/feature/primary__primary/object_features"
    
    var_name=a[path_features_names].value  
    var_name=[name[0] for name in var_name]    
    
    b=a[path_features]
    n,p=b.shape
    mat_features=np.zeros(shape=(n,p+1))
    mat_features[0:n,0:p]=b[0:n,0:p]
    mat_features[:,p]=list(range(n))

    mat_features=pd.DataFrame(mat_features)
    mat_features.columns=var_name+["line_id"]
    
    b=a[path_id]
    num_to_id={}
    id_to_num={}
    i=0
    for element in b[0:n]:
        num_to_id[i]=tuple(element)
        id_to_num[num_to_id[i]]=i
        i=i+1
    
    b=a[path_classification]
    y_pred=np.zeros(shape=(n,2))    
    for i in range(n):
        y_pred[i,0]=i
        y_pred[i,1]=b[i][0]
    y_pred=pd.DataFrame(y_pred)
    y_pred.columns=["line_id","y_hdf5"]
    mat_features=pd.merge(mat_features,y_pred, how='left', on="line_id")
    return(Connexions,trajectories,mat_features,num_to_id,id_to_num,movie_length)


def FilterTraj(trajectories,Connexions,threshold,mat_features,final_mat,id_to_num,Well,movie_length,length_threshold):
    for traj_n in range(len(trajectories)):
        s=Score(traj_n,trajectories,Connexions,id_to_num,movie_length,Well,mat_features)
        if ((s[0]>threshold)&(s[1]>threshold)):
            traj=trajectories[traj_n]
            points=traj.lstPoints
            n=len(points)
            
            if n>length_threshold:
                list_line=[]
                list_frame=[]
                for pt in points:
                    list_line.append(id_to_num[pt])
                    list_frame.append(pt[0])
                mat_features.loc[list_line,"Well"]=Well
                mat_features.loc[list_line,"traj"]=traj_n
                mat_features.loc[list_line,"Frame"]=list_frame
    new_mat=mat_features.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    if final_mat is None:
        final_mat=new_mat
        n_0=final_mat.shape[0]
        final_mat.index=range(n_0)
    else:
        n_0=final_mat.shape[0]
        n_1=new_mat.shape[0]
        new_mat.index=range(n_0,n_1+n_0)
        final_mat=pd.concat([final_mat,new_mat])
    return(final_mat)

class MitoCheck_Read():
    def __init__(self,file_location="./NegativeControl",threshold=0,length_threshold=0):
        print("MitoCheck_Read()")
        self.data=None
        self.Dir=[]
        for fn in os.listdir(file_location):
            self.Dir.append(fn)
            Well=fn.split("_")[-1]
            for fn_bis in os.listdir(file_location+"/"+fn):
                if  "traj" in fn_bis:
                    file_pkl=file_location+"/"+fn+"/"+fn_bis
                else:
                    file_hdf5=file_location+"/"+fn+"/"+fn_bis
            Connexions,trajectories,mat_features,num_to_id,id_to_num,movie_length=File(Well,file_pkl,file_hdf5) ### features with a lot of missing values
            self.data=FilterTraj(trajectories,Connexions,threshold,mat_features,self.data,id_to_num,Well,movie_length,length_threshold) ## filtering only the good trajectories
        p=self.data.shape[1]
        self.names=self.data.columns[0:(p-5)]
        self.Var_missing=self.names[[62,92,122,152]]
        features1=[2,4,5,6,8,9,16,17,18,23]
        features3=[31,32,33,34,35,37,42]
        features2=[24,25,26,27,28,29,30,62,92,122,152]
        features4=[0,3,153,162,164,217,218,219,220,221,237,238]
        features=features1+features2+features3+features4            
        self.error_names=self.data.columns[features]
        self.update()
    def Normalize(self,division=False,average=False):
        self.data=self.data.sort_values(["Well","traj","Frame"])
        grouped=self.data.groupby(['Well','traj'])
        if division and not average:
            print "Because of missing values I adjusted it, in the case X_t/X_0"
            #To prevent 0/0
            self.add_error()
            
        if average:
            def g(x,names):
                X_0=x[names].mean()
                return(X_0)
        else:
            def g(x,names):
                X_0=x.ix[x.index[0],names]
                return(X_0)
        if division:
            def f(x,names):        
                X_0=g(x,names)
                x[names]=x[names]/X_0
                return(x)
        else:
            def f(x,names):
                X_0=g(x,names)
                x[names]=x[names]-X_0
                return(x)
        self.data=grouped.apply(lambda x: f(x,self.names))
    def add_error(self):
        self.error_names=[el for el in self.error_names if el in self.data.columns]
        self.data.ix[self.data.index,self.error_names]+=1
    def missing_features(self):
        for name in self.Var_missing:
            if name in self.data.columns:
                self.data = self.data.drop(name, 1)
        self.names=[el for el in self.names if el not in self.Var_missing]
    def update(self,show=True):
        self.Group_of_traj=self.data.groupby(['Well','traj'])
        if show:
            print "Updated member Group_of_traj"
        

        
#s=MitoCheck_Read()
#s.Normalize()
