# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 17:15:42 2018

@author: DEREK
"""
import numpy as np
from numpy.linalg import eig
import scipy.linalg 
import os

if __name__ == '__main__':
    Max_Rshape_p5 = np.load('Max_Rshape_p5.npy')
    Ave_Rshape_p5 = np.load('Ave_Rshape_p5.npy')
    Std_Rshape_p5 = np.load('Std_Rshape_p5.npy')
    NameList = np.load('NameList.npy')
    N = np.shape(Max_Rshape_p5)[1]
    FILEA = 'D:/LM120K/Matching/PairA'
    FILEB = 'D:/LM120K/Matching/PairB'
    FILEB_N = 'D:/LM120K/NonMatching/PairB'

    K = len(Max_Rshape_p5)
    
    Max_ResultM = np.mean(Max_Rshape_p5,axis = 0)
    Ave_ResultM = np.mean(Ave_Rshape_p5,axis = 0)
    Std_ResultM = np.mean(Std_Rshape_p5,axis = 0)
    
    np.save('Ave_ResultM.npy',Ave_ResultM)
    np.save('Std_ResultM.npy',Std_ResultM)
    np.save('Max_ResultM.npy',Max_ResultM)
    
    Ave_ResultM =  np.load('Ave_ResultM.npy')
    Std_ResultM =  np.load('Std_ResultM.npy')
    Max_ResultM =  np.load('Max_ResultM.npy')
    
    Max = np.zeros([N,N])
    Ave = np.zeros([N,N])
    Std = np.zeros([N,N])

    
    Max_Rshape = []
    Ave_Rshape = []
    Std_Rshape = []
    
    for j in range(K):
        Max_Rshape_p5_1= np.reshape((Max_Rshape_p5[j]- Max_ResultM),(512,1))
        max_S_B = np.sqrt(np.sum(Max_Rshape_p5_1**2))
        max_norm_B = Max_Rshape_p5_1/max_S_B
        max_norm_B = np.nan_to_num(max_norm_B)
        Max_Rshape.append(max_norm_B)
        M = np.reshape(max_norm_B,(N,1))        
        M_T = np.transpose(M)
        Max_Cov = np.dot(M,M_T)
        Max= Max + Max_Cov
            
        Ave_Rshape_p5_1=np.reshape((Ave_Rshape_p5[j]- Ave_ResultM),(512,1))
        ave_S_B = np.sqrt(np.sum(Ave_Rshape_p5_1**2))
        ave_norm_B = Ave_Rshape_p5_1/ave_S_B
        ave_norm_B = np.nan_to_num(ave_norm_B)
        Ave_Rshape.append(ave_norm_B)
        A = np.reshape(ave_norm_B,(N,1))
        A_T = np.transpose(A)
        Ave_Cov = np.dot(A,A_T)
        Ave= Ave + Ave_Cov
            
        Std_Rshape_p5_1=np.reshape((Std_Rshape_p5[j]- Std_ResultM),(512,1))
        std_S_B = np.sqrt(np.sum(Std_Rshape_p5_1**2))
        std_norm_B = Std_Rshape_p5_1/std_S_B
        std_norm_B = np.nan_to_num(std_norm_B)
        Std_Rshape.append(std_norm_B)
        S = np.reshape(std_norm_B,(N,1))
        S_T = np.transpose(S)
        Std_Cov = np.dot(S,S_T)
        Std= Std + Std_Cov

    Max_Result= Max/(K-1)
    Ave_Result= Ave/(K-1)
    Std_Result= Std/(K-1)
        
    np.save('Max_Result.npy',Max_Result)
    np.save('Ave_Result.npy',Ave_Result)
    np.save('Std_Result.npy',Std_Result)
    
    np.save('Max_Rshape.npy',Max_Rshape)
    np.save('Ave_Rshape.npy',Ave_Rshape)
    np.save('Std_Rshape.npy',Std_Rshape)
    
    Max_Result =  np.load('Max_Result.npy')
    Ave_Result =  np.load('Ave_Result.npy')
    Std_Result =  np.load('Std_Result.npy')
        

    MPair1 = []
    MPair2 = []
    NPair1 = []
    NPair2 = []
    A = os.listdir(FILEA)
    B = os.listdir(FILEB)
    C = os.listdir(FILEB_N)
    for i in range(len(A)):
        print(i)
        KK = os.listdir(FILEA + '/' + A[i])
        QQ = os.listdir(FILEB + '/' + B[i])
        JJ = os.listdir(FILEB_N + '/' + C[i])
        for j in range(len(KK)):
            MA = KK[j]
            MB = QQ[j]
            NB = JJ[j]
            for k in range(2):
                if k == 0:
                    MPair1.append(MA)
                    MPair2.append(MB)
                    NPair1.append(MA)
                    NPair2.append(NB)
                if k == 1:
                    MPair1.append(MB)
                    MPair2.append(MA)
                    NPair1.append(NB)
                    NPair2.append(MA)
        
    Max_lambdam = []  
    Max_lambdan = []   
    Ave_lambdam = []  
    Ave_lambdan = []    
    Std_lambdam = []  
    Std_lambdan = []

    
    Max_Cm = []
    Max_Cn = []
    Ave_Cm = []
    Ave_Cn = []
    Std_Cm = []
    Std_Cn = []

    
    Max_MEV = []
    Ave_MEV = []
    Std_MEV = []

    
    NameList= NameList.tolist()
    
    Max_lambdao = np.zeros((N,N)) 
    Max_lambdai = np.zeros((N,N))
        
    Ave_lambdao = np.zeros((N,N)) 
    Ave_lambdai = np.zeros((N,N))
        
    Std_lambdao = np.zeros((N,N)) 
    Std_lambdai = np.zeros((N,N))
    
    
    for j in range(K):
        
        print(j)
        MA_IDX = NameList.index(MPair1[j])
        MB_IDX = NameList.index(MPair2[j])
            
        Max_MA = np.reshape(Max_Rshape[MA_IDX],(N,1))
        Max_MB = np.reshape(Max_Rshape[MB_IDX],(N,1))
        Max_T_MB = np.transpose(Max_MB)
        Max_MAB = np.dot(Max_MA,Max_T_MB) 
        Max_lambdao = Max_lambdao + Max_MAB 
            
        Ave_MA = np.reshape(Ave_Rshape[MA_IDX],(N,1))
        Ave_MB = np.reshape(Ave_Rshape[MB_IDX],(N,1))
        Ave_T_MB = np.transpose(Ave_MB)
        Ave_MAB = np.dot(Ave_MA,Ave_T_MB) 
        Ave_lambdao = Ave_lambdao + Ave_MAB # lambdao[49x49
            
        Std_MA = np.reshape(Std_Rshape[MA_IDX],(N,1))
        Std_MB = np.reshape(Std_Rshape[MB_IDX],(N,1))
        Std_T_MB = np.transpose(Std_MB)
        Std_MAB = np.dot(Std_MA,Std_T_MB) 
        Std_lambdao = Std_lambdao + Std_MAB 
            
        NA_IDX = NameList.index(NPair1[j])
        NB_IDX = NameList.index(NPair2[j])
            
        Max_NA = np.reshape(Max_Rshape[NA_IDX],(N,1))
        Max_NB = np.reshape(Max_Rshape[NB_IDX],(N,1))
        Max_T_NB = np.transpose(Max_NB)
        Max_NAB = np.dot(Max_NA,Max_T_NB) 
        Max_lambdai = Max_lambdai + Max_NAB 
            
        Ave_NA = np.reshape(Ave_Rshape[NA_IDX],(N,1))
        Ave_NB = np.reshape(Ave_Rshape[NB_IDX],(N,1))
        Ave_T_NB = np.transpose(Ave_NB)
        Ave_NAB = np.dot(Ave_NA,Ave_T_NB) 
        Ave_lambdai = Ave_lambdai + Ave_NAB 
            
        Std_NA = np.reshape(Std_Rshape[NA_IDX],(N,1))
        Std_NB = np.reshape(Std_Rshape[NB_IDX],(N,1))
        Std_T_NB = np.transpose(Std_NB)
        Std_NAB = np.dot(Std_NA,Std_T_NB) 
        Std_lambdai = Std_lambdai + Std_NAB 
            
    Max_lambdao = Max_lambdao/(K-1)
    Max_lambdai = Max_lambdai/(K-1) 
    Max_AA_1_2 = scipy.linalg.fractional_matrix_power(Max_Result, -0.5)
    Max_Mpi = np.dot(np.dot(Max_AA_1_2,Max_lambdao),Max_AA_1_2)
    Max_Npi = np.dot(np.dot(Max_AA_1_2,Max_lambdai),Max_AA_1_2)
    Max_Mw,Max_Mv = eig(Max_Mpi)
    Max_MEV.append(Max_Mv)
    Max_lambdam.append(Max_lambdao)
    Max_lambdan.append(Max_lambdai)
    Max_cm = np.dot(np.dot(np.transpose(Max_Mv),Max_Mpi),Max_Mv).diagonal()
    Max_cn = np.dot(np.dot(np.transpose(Max_Mv),Max_Npi),Max_Mv).diagonal()
    Max_Cm.append(Max_cm)
    Max_Cn.append(Max_cn)
    
    Ave_lambdao = Ave_lambdao/(K-1)
    Ave_lambdai = Ave_lambdai/(K-1) 
    Ave_AA_1_2 = scipy.linalg.fractional_matrix_power(Ave_Result, -0.5)
    Ave_Mpi = np.dot(np.dot(Ave_AA_1_2,Ave_lambdao),Ave_AA_1_2)
    Ave_Npi = np.dot(np.dot(Ave_AA_1_2,Ave_lambdai),Ave_AA_1_2)
    Ave_Mw,Ave_Mv = eig(Ave_Mpi)
    Ave_MEV.append(Ave_Mv)
    Ave_lambdam.append(Ave_lambdao)
    Ave_lambdan.append(Ave_lambdai)
    Ave_cm = np.dot(np.dot(np.transpose(Ave_Mv),Ave_Mpi),Ave_Mv).diagonal()
    Ave_cn = np.dot(np.dot(np.transpose(Ave_Mv),Ave_Npi),Ave_Mv).diagonal()
    Ave_Cm.append(Ave_cm)
    Ave_Cn.append(Ave_cn)
        
    Std_lambdao = Std_lambdao/(K-1)
    Std_lambdai = Std_lambdai/(K-1) 
    Std_AA_1_2 = scipy.linalg.fractional_matrix_power(Std_Result, -0.5)
    Std_Mpi = np.dot(np.dot(Std_AA_1_2,Std_lambdao),Std_AA_1_2)
    Std_Npi = np.dot(np.dot(Std_AA_1_2,Std_lambdai),Std_AA_1_2)
    Std_Mw,Std_Mv = eig(Std_Mpi)
    Std_MEV.append(Std_Mv)
    Std_lambdam.append(Std_lambdao)
    Std_lambdan.append(Std_lambdai)
    Std_cm = np.dot(np.dot(np.transpose(Std_Mv),Std_Mpi),Std_Mv).diagonal()
    Std_cn = np.dot(np.dot(np.transpose(Std_Mv),Std_Npi),Std_Mv).diagonal()
    Std_Cm.append(Std_cm)
    Std_Cn.append(Std_cn)
    

    np.save('Max_MEV.npy',Max_MEV)
    np.save('Max_Cm.npy',Max_Cm)
    np.save('Max_Cn.npy',Max_Cn)

    np.save('Ave_MEV.npy',Ave_MEV)
    np.save('Ave_Cm.npy',Ave_Cm)
    np.save('Ave_Cn.npy',Ave_Cn)

    np.save('Std_MEV.npy',Std_MEV)
    np.save('Std_Cm.npy',Std_Cm)
    np.save('Std_Cn.npy',Std_Cn)
    


    