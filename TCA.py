### All the packages we need:


import scipy as sp
from scipy.spatial.distance import pdist,squareform

def Gaussian(X,gamma,p):  ## computes matrix K
	R=pdist(X, metric='euclidean', p=p, w=None, V=None, VI=None)
	n=X.shape[0]
	def f(x):
		return(sp.exp(-(x**p)/(2*gamma**2)))
	res=map(f,R)
	ds=squareform(res)
	I=sp.zeros(shape=(n,n))
	sp.fill_diagonal(I,f(0))
	return(ds+I)

def TCA(X_S,X_T,m,mu,gamma=1,p=2,random_sample_T=1):
    
    X_S=sp.mat(X_S)
    X_T=sp.mat(X_T)
    
    n_S=X_S.shape[0]
    n_T=X_T.shape[0]
    
    if random_sample_T!=1:
        print str(int(n_T*random_sample_T))+" samples taken from the task domain"
        index_sample=sp.random.choice([i for i in range(n_T)],size=int(n_T*random_sample_T))
        X_T=X_T[index_sample,:]
        
        n_T=X_T.shape[0]
        

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
    
    K=Gaussian(sp.vstack([X_S,X_T]),gamma,p)
    n=K.shape[0]
    Id=sp.zeros(shape=(n,n))
    H=sp.zeros(shape=(n_S+n_T,n_S+n_T))
    sp.fill_diagonal(Id,1)
    sp.fill_diagonal(H,1)
    H-=1./(n_S+n_T)
    
    Id=sp.mat(Id)
    H=sp.mat(H)
    K=sp.mat(K)
    L=sp.mat(L)
    matrix_inv=Id+mu*K*L*K
    matrix_inv=sp.linalg.inv(matrix_inv)
    matrix=K*H*K
    matrix=matrix_inv*sp.mat(matrix)
    
    eigen_values=sp.linalg.eig(matrix)

    eigen_val=eigen_values[0][0:m]
    eigen_vect=eigen_values[1][:,0:m]
    return(eigen_val,eigen_vect,K,sp.vstack([X_S,X_T]))










































import pdb

from scipy import linalg as LA
import pandas as pd
def Gaus_Dens(x,gamma,p):
    return(sp.exp(-(x**p)/(2*gamma**2)))


def kernel_estimation(x,x_i,gamma,p):
    if sp.sum(pd.isnull(x-x_i))!=0:
        pdb.set_trace()
    NORM=LA.norm(x-x_i,p)
    return Gaus_Dens(NORM,gamma,p)

def new_feature(data_used,W_eigenVectors,x,gamma,p,names):
    data_used["kernel_distance"]=data_used.apply(lambda x_i: kernel_estimation(x[names],x_i[names],gamma,p),axis=1)
    #pdb.set_trace()
    n=data_used.shape[0]
    vertical_vect=sp.zeros(shape=(n,1))
    vertical_vect[:,0]=sp.array(data_used["kernel_distance"])
    new_feat=sp.mat(W_eigenVectors.T)*vertical_vect
    return(sp.array(new_feat).flatten())



def getting_kernel_projection(data,data_used,m,W_eigenVectors,gamma=1,p=2):
    data_used.columns=data.columns
    names=data.columns
    new_feat=["TCA_"+str(i) for i in range(m)]
    data.apply(lambda x: new_feature(data_used,W_eigenVectors,x,gamma,p,names),axis=1)
    pdb.set_trace()
    data[new_feat]=data.apply(lambda x: new_feature(data_used,W_eigenVectors,x,gamma,p,names),axis=1)

    return(data)
"""

num_str="0015" 
## Well name
if os.path.isfile("H2b_data.csv"):
    print "The file existed so I loaded it."
    H2b = Traj_data(file_name="H2B_N_D_0.csv",pkl_traj_file="./Pkl_file") 

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

"""
