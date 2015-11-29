### All the packages we need:

from Reader import Reader
from Randomforest import RandomForest_Autotunner,plot_matrix,Measure,check_rotate
from Traj_creator import Traj_data

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os.path
import cPickle as pkl

import scipy as sp

import pdb

def Gaussian(X,gamma,p):  ## computes matrix K
	R=sp.spatial.distance.pdist(X, metric='euclidean', p=p, w=None, V=None, VI=None)
	n=X.shape[0]
	q = lambda i,j,n: n*j - j*(j+1)/2 + i - 1 - j
	def f(x):
		return(sp.exp(-gamma*(x**p)))
	res=map(f,R)
	ds=sp.spatial.distance.squareform(res)
	for i in xrange( 1, n ):
    		for j in xrange( i ):
        		assert ds[ i, j ] == res[ q( i, j, n ) ]
	I=sp.zeros(shape=(n,n))
	sp.fill_diagonal(I,f(0))
	return(ds+I)

def TCA(X_S,X_T,m,mu,random_sample_T=1):
        
    
    n_S=X_S.shape[0]
    n_T=X_T.shape[1]
    
    if random_sample_T!=1:    
        index_sample=sp.random.choice([i for i in range(n_T)],size=int(n_T*random_sample_T))
        X_T=X_T[index_sample,:]
        n_T=X_T.shape[1]

    if m>(n_S+n_T):
    	print("m is larger then n_S+n_T, so it has been changed")
    	m=n_S+n_T
    
    
    L=sp.zeros(shape=(n_S+n_T,n_S+n_T))
    L_SS=sp.ones(shape=(n_S,n_S))/(n_S**2)
    L_TT=sp.ones(shape=(n_T,n_T))/(n_T**2)
    L_ST=-sp.ones(shape=(n_S,n_T))/(n_S*n_T)
    L_TS=-sp.ones(shape=(n_T,n_S))/(n_S*n_T)
    
    L[0:n_S,0:n_S]=L_SS
    L[n_S:n_S+n_T,n_S:n_S+n_T]=L_TT
    L[n_S:n_S+n_T,0:n_S]=L_TS
    L[0:n_S,n_S:n_S+n_T]=L_ST

    K=Gaussian(sp.vstack(X_S,X_T),1,2)
    n=K.shape[0]
    I=sp.zeros(shape=(n,n))
    H=sp.zeros(shape=(n_S+n_T,n_S+n_T))
    sp.fill_diagonal(I,1)
    sp.fill_diagonal(H,1)
    H-=1./(n_S+n_T)
    
    I=sp.mat(I)
    H=sp.mat(H)
    K=sp.mat(K)
    L=sp.mat(L)
    
    matrix_inv=I+mu*K*L*K
    matrix_inv=sp.linalg.inv(matrix_inv)
    matrix=K*H*K
    matrix=matrix_inv*sp.mat(matrix)
    
    eigen_values=sp.linalg.eig(matrix)
    pdb.set_trace()
    
    ind=[n_S+n_T-1-i for i in range(m)]
    eigen_val=eigen_values[0][ind]
    eigen_vect=eigen_values[1][:,ind]
    
    return([eigen_val,eigen_vect])



num_str="0015" 
## Well name
if os.path.isfile("H2b_data.csv"):
    print "The file existed so I loaded it."
    H2b = Traj_data(file_name="H2b_data.csv",pkl_traj_file="/home/pubuntu/Documents/InternWork2/Pkl_file") 

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


