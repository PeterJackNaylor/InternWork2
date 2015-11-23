# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:50:16 2015

@author: Peter-Jack
"""

import h5py
import numpy as np
import pandas as pd
from util.listFileManagement import ArffReader

file_name="0015PCNA.hdf5"

def f_position(tup):
    return [tup[0],tup[1]]
def f_position_0(tup):
    return [tup[0]]



class Reader():
    def __init__(self):
        print "Reader constructed"
        
    def hdf5_read(self,file_name,line_id=True,channel='primary'):
        ###If you choose channel secondary it has to also have a primary channel, it uses the id of the primary channel
        a=h5py.File(file_name, "r+")
        
        first="/sample/0/plate/"+a["/sample/0/plate/"].keys()[0]+"/experiment/0/position/"
        self.well=a[first].keys()[0]
        first+=self.well
        path_features=first+"/feature/"
        path_names="/definition/feature/"
        if channel=='primary':
            path_names+=a[path_features].keys()[0]+"/object_features"
            path_features+=a[path_features].keys()[0]
            path_position=path_features+"/center"
            path_features+="/object_features"
        elif channel=='secondary':
            path_names+=a[path_features].keys()[1]+"/object_features"
            path_features+=a[path_features].keys()[1]
            path_position=path_features+"/center"
            path_features+="/object_features"
        path_id=first+"/object/"
        path_id+=a[path_id].keys()[0]
        b=a[path_features]
        n,p=b.shape
        c=a[path_id]

        self.id_just_opened=c
        c=np.matrix(map(f_position,c[0:n]))
        d=a[path_position]
        e=a[path_names]
        d=np.matrix(map(f_position,d[0:n]))
        self.names=np.array([e[i][0] for i in range(p)])
        data=pd.DataFrame(b[0:n,0:p],columns=self.names,dtype='float64')
        
        data[self.well+'_id_frame']=c[0:n,0]
        data[self.well+'_id_object']=c[0:n,1]

        data[self.well+'_pos_x']=np.array(d[0:n,0])
        data[self.well+'_pos_y']=np.array(d[0:n,1])
        if line_id:
            data[self.well+'_line_id']=range(n)
        self.data=data
        self.type="HDF5"
        self.file_name=file_name
        
        self.Var_missing=np.array(self.names)[[62,92,122,152]]

    def arrf_read(self,file_name):
        reader = ArffReader(file_name)
        data = None
        labels = []
            
        for el in reader.dctFeatureData:
            data=reader.dctFeatureData[el] if data is None else np.vstack((data, reader.dctFeatureData[el]))
            labels.extend([el for k in range(len(reader.dctFeatureData[el]))])
            
        da=pd.DataFrame(data,columns=reader.lstFeatureNames)
        da["label"]=labels
        self.data=da
        self.type="Arrf"
        self.file_name=file_name
        self.names=reader.lstFeatureNames
        self.Var_missing=np.array(self.names)[[62,92,122,152]]
    def renaming_for_mitosis(self):
        def f(name):
            if name in ['Interphase','Elongated','Large']:
                return('S')
            elif name in ['Metaphase', 'Anaphase', 'Prometaphase', 'MetaphaseAlignment']:
                return('M')
            else:
                return('O')
        self.data["label"]=self.data.apply(lambda r: f(r["label"]),axis=1)
    def renaming(self):
        def bij(val_string):
            if   val_string=="G1":
                return "1"
            elif val_string=="G2":
                return "2"
            elif val_string=="M":
                return "M"
            else:
                return "S"
        self.data["label"]=self.data.apply(lambda r: bij(r["label"]),axis=1)
    def missing_features_data(self):
        for name in self.Var_missing:
            if name in self.data.columns:
                self.data = self.data.drop(name, 1)
        self.names=[el for el in self.names if el not in self.Var_missing]    
"""
file_name="0015PCNA.hdf5"   
t=Reader()
t.hdf5_read(file_name)
data=t.data

file_name="PCNA_labels.arff"
s=Reader()
s.arrf_read(file_name)
data_2=s.data
"""

train_file="PCNA_labels.arff"
test_file="0015_PCNA.hdf5"
"""
train=Reader()
train.arrf_read(train_file)
test=Reader()
test.hdf5_read(test_file)
data=test.data
train.names
"""

"""
train_file="PCNA_labels.arff"
test_file="0015PCNA.hdf5"

train=Reader()
train.arrf_read(train_file)
train.renaming()
d=train.data
print d.head()

"""

