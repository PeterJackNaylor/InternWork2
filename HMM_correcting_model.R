HMM_correct=function(data_temp,EmissionMat,y_label,obs_number=0){
  print("We are here going to correct the data predicted in the first part  
         with a hidden markov where we infere the transition matrix with a 
         strong prior knowlegde in biology. A five state classifier :" )
  print("M_E -> G1 -> S -> G2 -> M_E")
  obs_number=0
  data=data_temp ## data_temp has to have a traj column and y_label
  library(HMM)
  trans=list()  #List of outputs
  emi=list()    
  
  emission_matrix=EmissionMat
  transition_prior=t(matrix(c(  .7  , .27 , .01 , .01 , .01
                                , .01 , .7  , .27 , .01 , .01
                                , .01 , .01 , .7  , .27 , .01
                                , .01 , .01 , .01 , .7  , .27
                                , .01 , .01 , .01 , .01  , 0.96 ),5)) 
  
  startProbs=c(0.3, 0.7, 0.0, 0.0, 0.0)
  
  hmm = initHMM(c("1","2","3","4","5"),c("1","2","3","4","5"),
                transProbs=transition_prior,
                emissionProbs=emission_matrix
                ,startProbs=startProbs)
  
  print(paste("To assess briefly what is happening, we printed a sequence of observation, sequence number:",toString(obs_number) ))
  n_traj=max(data$traj)
  j=1
  for (i in 0:(n_traj-1)){
    obs=data[data$traj==i,y_label]
    if (i==obs_number){
      test=obs
      print(test)
    }
    
    hmm = initHMM(c("1","2","3","4","5"),c("1","2","3","4","5"),
                  startProbs=startProbs,
                  emissionProbs=emission_matrix,
                  transProbs=transition_prior)
    bw=try(baumWelch(hmm,observation=obs,10),silent = TRUE)
    if (!(inherits(bw, "try-error"))){
      trans[[j]]=bw$hmm$transProbs
      emi[[j]]=bw$hmm$emissionProbs
      j=j+1  
    }
  }
  
  mean_t=matrix(0,ncol=5,nrow=5)
  mean_e=matrix(0,ncol=5,nrow=5)
  for (i in 1:length(trans)){
    mean_t=mean_t+trans[[i]]
    mean_e=mean_e+emi[[i]]
  }
  mean_t=mean_t/length(trans)
  mean_e=mean_e/length(trans)
  
  var_t=matrix(0,ncol=5,nrow=5)
  var_e=matrix(0,ncol=5,nrow=5)
  
  for (i in 1:length(trans)){
    var_t=var_t+(trans[[i]]-mean_t)*(trans[[i]]-mean_t)
    var_e=var_e+(emi[[i]]-mean_e)*(emi[[i]]-mean_e)
  }
  var_t=var_t/length(trans)
  var_e=var_e/length(trans)
  
  
  ligne1=mean_t[1,1]+mean_t[1,2]
  ligne2=mean_t[2,2]+mean_t[2,3]
  ligne3=mean_t[3,3]+mean_t[3,4]
  ligne4=mean_t[4,4]+mean_t[4,5]
  
  transProbs=t(matrix(c(mean_t[1,1]/ligne1  , mean_t[1,2]/ligne1    ,0                 , 0                 ,0
                        , 0                 , mean_t[2,2]/ligne2    ,mean_t[2,3]/ligne2, 0                 ,0
                        , 0                 , 0                     ,mean_t[3,3]/ligne3, mean_t[3,4]/ligne3,0
                        , 0                 , 0                     ,0                 , mean_t[4,4]/ligne4,mean_t[4,5]/ligne4
                        , 0                 , 0                     ,0                 , 0                 ,1 ),5))
  startProbs=c(0.3, 0.7, 0.0, 0.0, 0.0)
  
  hmm = initHMM(c("1","2","3","4","5"),c("1","2","3","4","5"),
                transProbs=transProbs,
                emissionProbs=emission_matrix  ##We still keep the confusion matrix of the first classification
                ,startProbs=startProbs)
  
  print(paste("To assess briefly what is happening, we printed the corrected sequence, number:",toString(obs_number) ))

  for (i in 0:(n_traj-1)){
    obs=data[data$traj==i,y_label]
    new_obs=as.integer(viterbi(hmm,observation=obs))
    data[data$traj==i,"HMM"]=new_obs
    if (i==obs_number){
      print(new_obs)
    }
  }
    
  return(list(new_data=data,mean_t=mean_t,mean_e=mean_e,var_t=var_t,var_e=var_e,transProbs=transProbs))
}




Predict_from_hmm=function(data,y_label,y_label_to_give,transProbs,emission_matrix,startProbs,obs_number=0){
  hmm = initHMM(c("1","2","3","4","5"),c("1","2","3","4","5"),
                transProbs=transProbs,
                emissionProbs=emission_matrix  ##We still keep the confusion matrix of the first classification
                ,startProbs=startProbs)
  i=0
  keys_=unique(data[c("Well","traj")])
  rows_of_keys <- rownames(keys_) 
  print(paste("To assess briefly what is happening, we printed the corrected sequence, number:",toString(obs_number) ))
  for (r in rows_of_keys){
    tup_data_frame=keys_[r,]
    w=tup_data_frame$Well
    t=tup_data_frame$traj
    obs=data[(data$traj==t)&(data$Well==w),y_label]
    if (sum(obs!="nan")>(length(obs)/2)){
      new_obs=as.integer(viterbi(hmm,observation=obs))
      data[(data$traj==t)&(data$Well==w),y_label_to_give]=new_obs
      
    }else{
      
      print("Only nan's here...id:")
      print(r)
    }
    if (i==obs_number){
      print(obs)
      print(new_obs)
      
    }
    i=i+1
  }
  return(data)
  
}
