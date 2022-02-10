import tensorflow as tf
import numpy as np
import os
import numpy.matlib
np.set_printoptions(precision=8)

            
if __name__ == '__main__':
    print('1stCCA_1')
    FILEA = 'D:/Landmarks/Matching/PairA'
    FILEB = 'D:/Landmarks/Matching/PairB'
    A = os.listdir(FILEA)
    B = os.listdir(FILEB)
   
    label_A = []    
    label_B = []            
    Channel = range(512)
    for i in range(len(A)):
        print(i)
        FolderA = A[i]
        FolderB = B[i]
        for j in range(len(os.listdir(FILEA+'/'+os.listdir(FILEA)[i]))):          
            FileA = os.listdir(FILEA+'/'+ FolderA)[j]
            FileB = os.listdir(FILEB +'/'+ FolderB)[j]
            label_A.append(int(FolderA))                     
            label_B.append(int(FolderB))
   
    label = np.concatenate((label_A, label_B), axis=0)
    np.save('label.npy',label)
    label = np.load('label.npy')    
    Laplcianmatrix = np.zeros((len(label),len(label)))    
    for i in range(len(label)):
        print(i)
        for j in range(len(label)):
            if label[i] == label[j] & i != j:
                Laplcianmatrix[i,j] = 0
            if label[i] != label[j] & i != j:
                Laplcianmatrix[i,j] = -1
            else:
                Laplcianmatrix[i,i] = -np.sum(Laplcianmatrix[:,i]+Laplcianmatrix[i,i])
                
    Laplcianmatrix = Laplcianmatrix - np.diag(np.sum(Laplcianmatrix,axis=0))
    Laplcianmatrix = np.nan_to_num(Laplcianmatrix)
                
    np.save('Laplcianmatrix.npy',Laplcianmatrix)
                
    
