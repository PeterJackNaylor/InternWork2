# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:43:42 2015

@author: Peter-Jack
"""

import warnings


def find_traj_mother(dicty,traj_n):
    found=False
    for k in dicty.keys():
        if traj_n in dicty[k]:
            found=True
            if ((len(dicty[k])==2)&(len(k)==1)):
                break
            else:
                k=(-1,)
                break ##It can be fusion, or merge
    if found:
        return(k)
    else:
        return((-1,)) 

def find_traj_sons(dicty,traj_n,Well):
    found=False
    for k in dicty.keys():
        if traj_n in k:
            found=True
            if ((len(dicty[k])==2)&(len(k)==1)):
                break
            else:
                k=(-1,)
                break ##It can be fusion, or merge
    if "k" not in locals():
        warnings.warn('No connexions in trajectory  '+str(traj_n)+'on well '+str(Well))
        k=(-1,)
    if (found&(k[0]!=-1)):
        return(dicty[k])
    else:
        return((-1,)) 


def Real_Mitosis(mother,list_son):
    score=0
    if mother in [6.0,9.0,8.0]:
        score+=1
    for son in list_son:
        if son==7.0:
            score+=1
    return(score)


def son_class(traj_n,trajectories,mat_features,id_to_num):
    traj=trajectories[traj_n]
    KEYS=sorted(traj.lstPoints,key=lambda x: x[0])
    line_son=id_to_num[KEYS[0]]
    return(mat_features.loc[line_son]["y_hdf5"])
    
    
def Score(traj_n,trajectories,Connexions,id_to_num,movie_length,Well,mat_features):
    traj=trajectories[traj_n]
    mother=None
    sons=[]
    KEYS=sorted(traj.lstPoints,key=lambda x: x[0])
    n_KEYS=len(KEYS)
    first_frame=KEYS[0][0]
    last_frame=KEYS[n_KEYS-1][0]
    line_son=id_to_num[KEYS[0]] 
    sons.append(mat_features.loc[line_son]["y_hdf5"])    
    
    if first_frame==0:
        mother=-1
    else:
        traj_mother_num=find_traj_mother(Connexions[first_frame-1],traj_n)
        if traj_mother_num[0]!=-1:
            traj_mother=trajectories[traj_mother_num[0]]
            
            KEYS=sorted(traj_mother.lstPoints,key=lambda x: x[0])
            n_KEYS=len(KEYS)
            id_mother=KEYS[n_KEYS-1]
            line_mother=id_to_num[id_mother]
            mother=mat_features.loc[line_mother]["y_hdf5"]
            
            if Connexions[first_frame-1][traj_mother_num][0]!=traj_n:
                Brother=Connexions[first_frame-1][traj_mother_num][0]
            else:
                Brother=Connexions[first_frame-1][traj_mother_num][1]
            
            traj_brother=trajectories[Brother]
            KEYS=sorted(traj_brother.lstPoints,key=lambda x: x[0])
            n_KEYS=len(KEYS)
            id_brother=KEYS[n_KEYS-1]
            line_brother=id_to_num[id_brother]
            sons.append(mat_features.loc[line_brother]["y_hdf5"])
        else:
            mother=-1
    if mother!=-1:
        score1=Real_Mitosis(mother,sons)
    else:
        score1=0
    
    if last_frame!=(movie_length-1):
        line_mother2=id_to_num[KEYS[n_KEYS-1]]
        mother2=mat_features.loc[line_mother2]["y_hdf5"]
        sons_traj=find_traj_sons(Connexions[last_frame],traj_n,Well)
        if sons_traj[0]!=-1:
            son0=son_class(sons_traj[0],trajectories,mat_features,id_to_num)
            son1=son_class(sons_traj[1],trajectories,mat_features,id_to_num)
        else:
            mother2=-1
    else:
        mother2=-1
    if mother2!=-1:
        score2=Real_Mitosis(mother2,[son0,son1])
    else:
        score2=0
    return((score1,score2))
    