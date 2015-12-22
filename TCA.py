### All the packages we need:


import scipy as sp
from scipy.spatial.distance import pdist,squareform

def Gaussian(R,kernel_para,p):  ## computes matrix K
    def f(x):
        return(sp.exp(-(x**p)/(2*kernel_para**2)))
    res=map(f,R)
    ds=squareform(res)	
    n=ds.shape[0]
    I=sp.zeros(shape=(n,n))
    sp.fill_diagonal(I,f(0))
    
    return(ds+I)

def Laplace(R,p,sigma):
    
    def f(x):
        return(sp.exp(-(x**p)/(2*sigma**2)))
    res=map(f,R)
    ds=squareform(res)
    n=ds.shape[0]
    I=sp.zeros(shape=(n,n))
    sp.fill_diagonal(I,f(0))
    M=ds+I
    d=sp.sum(M, axis=0)
    D=sp.diag(d)
    
    return(D-M)

def TCA(X_S,X_T,m=40,mu=0.1,kernel_para=1,p=2,random_sample_T=0.01):
    
    X_S=sp.mat(X_S)
    X_T=sp.mat(X_T)
    
    
    n_S=X_S.shape[0]
    n_T=X_T.shape[0]
    if random_sample_T!=1:
        print str(int(n_T*random_sample_T))+" samples taken from the task domain"
        index_sample=sp.random.choice([i for i in range(n_T)],size=int(n_T*random_sample_T))
        X_T=X_T[index_sample,:]
        
        n_T=X_T.shape[0]
    
    n=n_S+n_T         

    if m>(n):
        print("m is larger then n_S+n_T, so it has been changed")
        m=n
    

    L=sp.zeros(shape=(n,n))
    L_SS=sp.ones(shape=(n_S,n_S))/(n_S**2)
    L_TT=sp.ones(shape=(n_T,n_T))/(n_T**2)
    L_ST=-sp.ones(shape=(n_S,n_T))/(n_S*n_T)
    L_TS=-sp.ones(shape=(n_T,n_S))/(n_S*n_T)
    
    L[0:n_S,0:n_S]=L_SS
    L[n_S:n_S+n_T,n_S:n_S+n_T]=L_TT
    L[n_S:n_S+n_T,0:n_S]=L_TS
    L[0:n_S,n_S:n_S+n_T]=L_ST
    
    
    R=pdist(sp.vstack([X_S,X_T]), metric='euclidean', p=p, w=None, V=None, VI=None)

    K=Gaussian(R,kernel_para,p)

    Id=sp.zeros(shape=(n,n))
    H=sp.zeros(shape=(n,n))
    sp.fill_diagonal(Id,1)
    sp.fill_diagonal(H,1)
    H-=1./n

    Id=sp.mat(Id)
    H=sp.mat(H)
    K=sp.mat(K)
    L=sp.mat(L)
    
    matrix=sp.linalg.inv( K * L * K + mu * Id )*sp.mat( K * H * K )
    
    eigen_values=sp.linalg.eig(matrix)
    
    eigen_val=eigen_values[0][0:m]
    eigen_vect=eigen_values[1][:,0:m]
    return(eigen_val,eigen_vect,K,sp.vstack([X_S,X_T]))


def SSTCA(X_S,y_S,X_T,m=40,mu=0.1,lamb=0.0001,kernel_para=1,p=2,sigma=1,gamma=0.5,random_sample_T=0.01):
    
    X_S=sp.mat(X_S)
    X_T=sp.mat(X_T)
    
    y_S=sp.array(y_S)
    
    n_S=X_S.shape[0]
    n_T=X_T.shape[0]
    if random_sample_T!=1:
        print str(int(n_T*random_sample_T))+" samples taken from the task domain"
        index_sample=sp.random.choice([i for i in range(n_T)],size=int(n_T*random_sample_T))
        X_T=X_T[index_sample,:]
        
        n_T=X_T.shape[0]
    
    n=n_S+n_T         

    if m>(n):
        print("m is larger then n_S+n_T, so it has been changed")
        m=n
    

    L=sp.zeros(shape=(n,n))
    L_SS=sp.ones(shape=(n_S,n_S))/(n_S**2)
    L_TT=sp.ones(shape=(n_T,n_T))/(n_T**2)
    L_ST=-sp.ones(shape=(n_S,n_T))/(n_S*n_T)
    L_TS=-sp.ones(shape=(n_T,n_S))/(n_S*n_T)
    
    L[0:n_S,0:n_S]=L_SS
    L[n_S:n_S+n_T,n_S:n_S+n_T]=L_TT
    L[n_S:n_S+n_T,0:n_S]=L_TS
    L[0:n_S,n_S:n_S+n_T]=L_ST
    
    
    R=pdist(sp.vstack([X_S,X_T]), metric='euclidean', p=p, w=None, V=None, VI=None)

    K=Gaussian(R,kernel_para,p)

    Id=sp.zeros(shape=(n,n))
    H=sp.zeros(shape=(n,n))
    sp.fill_diagonal(Id,1)
    sp.fill_diagonal(H,1)
    H-=1./n

    LA=Laplace(R,p,sigma)    

    K_hat_y=sp.zeros(shape=(n,n))
    K_hat_y[0,0]=1
    for i in range(1,n_S):
        K_hat_y[i,i]=1
        for j in range(i):
            if y_S[i]==y_S[j]:
                K_hat_y[i,j]=1
                K_hat_y[j,i]=1
                
    K_hat_y=gamma*K_hat_y+(1-gamma)*Id
    Id=sp.mat(Id)
    H=sp.mat(H)
    K=sp.mat(K)
    L=sp.mat(L)
    LA=sp.mat(LA)
    
    matrix=sp.linalg.inv( K * (L + lamb*LA) * K + mu * Id )*sp.mat( K * H * K_hat_y * H * K )
    
    eigen_values=sp.linalg.eig(matrix)
    
    eigen_val=eigen_values[0][0:m]
    eigen_vect=eigen_values[1][:,0:m]
    return(eigen_val,eigen_vect,K,LA,K_hat_y,sp.vstack([X_S,X_T]))
















import pdb

from scipy import linalg as LA
import pandas as pd
def Gaus_Dens(x,kernel_para,p):
    return(sp.exp(-(x**p)/(2*kernel_para**2)))


def kernel_estimation(x,x_i,kernel_para,p):
    if sp.sum(pd.isnull(x-x_i))!=0:
        pdb.set_trace()
    NORM=LA.norm(x-x_i,p)
    return Gaus_Dens(NORM,kernel_para,p)

def new_feature(data_used,W_eigenVectors,x,kernel_para,p,names):
    data_used["kernel_distance"]=data_used.apply(lambda x_i: kernel_estimation(x[names],x_i[names],kernel_para,p),axis=1)
    n=data_used.shape[0]
    vertical_vect=sp.zeros(shape=(n,1))
    vertical_vect[:,0]=sp.array(data_used["kernel_distance"])
    new_feat=sp.mat(W_eigenVectors.T)*vertical_vect
    return(sp.array(new_feat).flatten())



def getting_kernel_projection(data,data_used,m,W_eigenVectors,kernel_para=1,p=2):
    names=sp.array(data.columns) 
    data_used=pd.DataFrame(data_used,columns=names)
    new_feat=[i for i in range(m)]
    data[new_feat]=data.apply(lambda x: new_feature(data_used,W_eigenVectors,x,kernel_para,p,names),axis=1)

    return(data)
"""

num_str="0015" 
## Well name
if os.path.isfile("H2b_data.csv"):
    print "The file existed so I loaded it."
    H2b = Traj_data(file_name="H2B_N_D_0.csv",pkl_traj_file="./Pkl_file") 
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
