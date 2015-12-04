# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 13:43:57 2015

@author: naylor
"""

import os 
import cPickle as pkl
from Reader import Reader
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


class Traj_data:
    def __init__(self,file_name=None,pkl_traj_file='/home/naylor/Documents/Work/Files/pkl'):
        print("Traj_data built")


        local_dir_hdf5=pkl_traj_file
        local_dir_pkl=local_dir_hdf5
        
        
        self.files_hdf5=[]
        for fn in os.listdir(local_dir_hdf5):
            if 'hdf5' in fn:
                self.files_hdf5.append(local_dir_hdf5+"/"+fn)
                
        self.files_pkl=[]
        for fn in os.listdir(local_dir_pkl):
            if 'pkl' in fn:
                self.files_pkl.append(local_dir_pkl+"/"+fn)
        if file_name is not None:
            self.extracting("0015","both_channels_0015.hdf5",'primary')
            self.data=pd.read_csv(file_name)
            self.update()            
            
    def extracting(self,num_str,file_loc_hdf5=None,channel='primary'):
        File_num_pkl=[el for el in self.files_pkl if num_str in el]
        File_num_hdf5=[el for el in self.files_hdf5 if num_str in el]
        for el in File_num_pkl:
            if "cycle_cens" in el:
                file_loc=el
                
        fp=open(file_loc,'r')
        a=pkl.load(fp)
        fp.close()
    
        right_traj_ind=a['length'].keys()    
    
        ## traj_noF_densities
    
        for el in File_num_pkl:
            if "traj_intQC" in el:
                file_loc=el
             
        fp=open(file_loc,'r')
        a=pkl.load(fp)
        fp.close()
    
        first=a[a.keys()[0]].keys()[0]
        second=a[a.keys()[0]][first].keys()[0]
    
        a_1=a[a.keys()[1]][first][second]
        
        self.trajectories=[a_1.lstTraj[ind] for ind in range(len(a_1.lstTraj)) if ind in right_traj_ind]
        self.all_trajectories=[a_1.lstTraj[ind] for ind in range(len(a_1.lstTraj))]        
        
        if file_loc_hdf5 is None:
            file_loc=File_num_hdf5[0]
        else:
            file_loc=file_loc_hdf5

        self.hdf5_reader=Reader()
        self.hdf5_reader.hdf5_read(file_loc,line_id=True,channel=channel)
        self.names=self.hdf5_reader.names
        
        self.data=self.hdf5_reader.data
        
        self.Var_missing=self.hdf5_reader.names[[62,92,122,152]]

        c=self.hdf5_reader.id_just_opened
        n,=c.shape
        self.mat_id=c[0:n]
        def id_t(x):
            return(tuple(x))
        self.mat_id=map(id_t,self.mat_id)
            
        self.mat_id_inv={}
        for i in range(len(self.mat_id)):
             self.mat_id_inv[self.mat_id[i]]=i
             
    def missing_features_data(self):
        for name in self.Var_missing:
            if name in self.data.columns:
                self.data = self.data.drop(name, 1)
        self.names=[el for el in self.names if el not in self.Var_missing]
    def missing_features_train(self):
        for name in self.Var_missing:
            if name in self.train.columns:
                self.train = self.train.drop(name, 1)
        self.names=[el for el in self.names if el not in self.Var_missing]

    def add_error(self):
        features1=[2,4,5,6,8,9,16,17,18,23]
        features3=[31,32,33,34,35,37,42]
        features2=[24,25,26,27,28,29,30,62,92,122,152]
        features4=[0,3,153,162,164,217,218,219,220,221,237,238]
        features=features1+features2+features3+features4
        self.data.ix[self.data.index,self.data.columns[features]]+=1
        
    def label_finder(self,file_name):
        file_loc="D:/cellcog/for cell cognition/classifier/annotations"+"/PLLT0001_01___P0015___T00001.xml"
        file_loc=file_name
    ##    file_loc="D:/cellcog/pcna_eth/classifier/three_phases/annotations/PLPlate1___P0015___T00001_bis.xml"
        tree = ET.parse(file_loc)
        root = tree.getroot()

        data_0015=np.zeros(shape=(2000,4))
        
        seq=0
        for i in range(len(root[1])):
            if len(root[1][i])!=0 and len(root[1][i])!=1:
                for j in range(len(root[1][i])):
                    if len(root[1][i][j])==0:
                        Type=root[1][i][j].text
                    else:
                        data_0015[seq,:]=[Type,root[1][i][j][0].text,root[1][i][j][1].text,root[1][i][j][2].text]
                        seq=seq+1
        for i in range(len(data_0015)):
            if data_0015[i,1]==0:
                break
        data_0015=data_0015[0:i,:]
        data_0015=pd.DataFrame(data_0015)
        data_0015.columns=["Type","x_c","y_c","time_idx"]
        full_data_0015=self.data[[self.hdf5_reader.well+"_id_frame",self.hdf5_reader.well+"_pos_x",self.hdf5_reader.well+"_pos_y"]]
        full_data_0015.columns=["time_idx","x","y"]
        full_data_0015["Type"]=0
        
        for frame in set(list(data_0015["time_idx"])):
            A_f=data_0015[data_0015["time_idx"]==frame]        
            B_f=full_data_0015[full_data_0015["time_idx"]==frame]
            for A_line in A_f.index:
                x_c=A_f.loc[A_line]["x_c"]
                y_c=A_f.loc[A_line]["y_c"]
                B_f_temp=B_f
                B_f_temp["Distance"]=(B_f_temp["x"]-x_c)**2+(B_f_temp["y"]-y_c)**2
                min_ind=B_f_temp["Distance"].idxmin(axis=1)
                full_data_0015.ix[min_ind,"Type"]=A_f.loc[A_line]["Type"]
        self.labels_and_line=full_data_0015[full_data_0015["Type"]!=0]
        self.labels_and_line.columns=[self.hdf5_reader.well+"_id_frame",self.hdf5_reader.well+"_pos_x",self.hdf5_reader.well+"_pos_y","Type"]
        
    def renaming_and_merge(self):
        def bij(val_string):
            val_string=int(val_string)
            if   val_string==1:
                return "1"
            elif val_string==2:
                return "S"
            elif val_string==3:
                return "S"
            elif val_string==4:
                return "S"
            elif val_string==5:
                return "2"
            else:
                return "M"
        self.labels_and_line["Type"]=self.labels_and_line.apply(lambda r: bij(r["Type"]),axis=1)
        self.data = self.data.join(self.labels_and_line["Type"])
        self.train=self.data[pd.notnull(self.data["Type"])]

    def Add_traj(self,normalize=False,all_traj=False,average=False,diff=False,num_traj=0):
## It can be improved with a grouby and lambda function (once they have traj
        if all_traj:
            traj_dic=self.all_trajectories
        else:
            traj_dic=self.trajectories
            
        if num_traj!=0:
            traj_dic=[traj_dic[i] for i in range(num_traj)]
        i=0
        for traj in traj_dic:
    
            list_feat=[]        
            for key in traj.lstPoints.keys():
                if key in self.mat_id_inv.keys():
                    list_feat.append(self.mat_id_inv[key])
                else:
                    print key
                    print "this is not the best signe..., maybe wrong xml file or wrong hdf5, or wrong traj"
            list_feat.sort()

            if normalize:
                if average:
                    X_nor=self.data[self.names].mean(axis=0)
                else:
                    X_nor=self.data.ix[list_feat[0],self.names]
                if diff:
                    X_=self.data.ix[list_feat,self.names] - X_nor
                else:
                    X_=self.data.ix[list_feat,self.names] / X_nor
                self.data.ix[list_feat,self.names]=X_
            
            self.data.ix[list_feat,"traj"]=i
            i+=1
        self.Group_of_traj=self.data.groupby('traj')
        first_word="Normalized" if normalize else "Unnormalzied"
        second_word="Averaged" if average else ""
        if normalize:
            third_word="Subtracted" if diff else "Divided"
        else:
            third_word=""
        self.caract=first_word+"_"+second_word+"_"+third_word
    def update(self,show=True):
        self.Group_of_traj=self.data.groupby('traj')
        if show:
            print "Updated member Group_of_traj"
        we="0015"
        self.labels_and_line=self.data[[we+"_id_frame",we+"_pos_x",we+"_pos_y","Type"]]
        self.labels_and_line=self.labels_and_line[pd.notnull(self.labels_and_line['Type'])]
        self.train=self.data[pd.notnull(self.data["Type"])]
    def filter_length_traj(self,mu):
        new_data=self.data.groupby('traj').filter(lambda x: len(x) >= mu)
        self.data=new_data
        self.update(show=False)

##test=Traj_data(file_name="PCNA_data.csv")
"""

t=Traj_data()
t.extracting(num_str,"both_channels_0015.hdf5",'secondary')
t.Add_traj()
file_loc="0015_PCNA.xml"
t.label_finder(file_loc)
d=t.data

s=t.data
#lab=t.labels_and_line
test=Reader()
test.hdf5_read("0015_PCNA.hdf5")

test2=Reader()
test2.hdf5_read("0015_PCNA_with_h2b_cut.hdf5",line_id=True,channel='secondary')
sss=test2.data

"""
"""
## Well name
num_str="0015"
if os.path.isfile("H2B_N_F_A_test.csv"):
    print "The file existed so I loaded it."
    H2B_N_F_A = Traj_data(file_name="H2B_N_F_A_test.csv")#,pkl_traj_file="/home/pubuntu/Documents/InternWork2/Pkl_file") 
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

    H2B_N_F_A.data.to_csv('H2B_N_F_A_test.csv',index=False,header=True)    
"""
