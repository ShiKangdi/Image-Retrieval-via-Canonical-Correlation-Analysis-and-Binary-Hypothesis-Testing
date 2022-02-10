import numpy as np
import os
import scipy.linalg 
from sympy import Symbol
from numpy.linalg import inv,det
import math as m
from numpy.linalg import eig



METHOD = 'Std'
Laplcianmatrix = np.load('Laplcianmatrix.npy')
T_Max_Rshape_p5 = np.load(METHOD + '_Rshape.npy')
N = np.shape(T_Max_Rshape_p5)[1]
K = len(T_Max_Rshape_p5)
T_Max_Rshape_p5 = np.reshape(T_Max_Rshape_p5,(K,512))
Xor = np.dot(np.dot(np.transpose(T_Max_Rshape_p5),Laplcianmatrix),T_Max_Rshape_p5)
Eigval,Eigvec = eig(Xor)
TrEigval = np.sort(Eigval)[::-1]
EigvecInd = np.argsort(Eigval)[::-1]
TrEigvec = Eigvec[:,EigvecInd]
np.save('Max_Eigenvec.npy',TrEigvec)
np.save('Max_Eigenval.npy',TrEigval)

       
if __name__ == '__main__':
    print('Evaluation_2')
    
 
    Ave_Rshape_p5 = np.load('Eva_'+ METHOD +'_Rshape_p5.npy')
    NameList = np.load('Eva_NameList.npy')
    
    e = [512,450,400,300,200,100,50,25]
    for n_comp in e: 
        All_S_Transformed = []
        PCA = np.dot(np.diag(1/np.sqrt(TrEigval[:n_comp])),np.transpose(TrEigvec[:,:n_comp]))
        for mm in range(len(Ave_Rshape_p5)):
            print(mm)
            Max_DPB = np.dot(PCA,Ave_Rshape_p5[mm])
            Max_DPB = np.nan_to_num(Max_DPB)
            Max_S_B = np.sqrt(np.sum(Max_DPB**2))
            Max_norm_B = Max_DPB/Max_S_B
            Max_norm_B = np.nan_to_num(Max_norm_B)
            All_S_Transformed.append(Max_norm_B)
        np.save('Eva_Trans_Max_Rshape_P5.npy',All_S_Transformed)  
        
        Query = All_S_Transformed[:55]
        All = All_S_Transformed[55:]
        All_Name = NameList[55:]
        np.save('Sec_All.npy',All)
        np.save('Sec_Query.npy',Query)
        np.save('Sec_All_Name.npy',All_Name)
        
        O = len(All)
        for q in range(55):
            dic = {}
            X = Query[q]
            for k in range(O):
                print([q,k])
                Y = All[k]
                Xi = np.reshape(X,(n_comp,1))
                Yi = np.reshape(Y,(n_comp,1))          
                P = np.dot(np.transpose(Xi),Yi)                       
                dic[All_Name[k][:-4]] = P
            b = sorted(dic.items(), key=lambda d: d[1],reverse=True)
            if not os.path.exists("D:/2022/3W/Oxford/SPCA_3/" + METHOD  + str(n_comp)):
                os.makedirs("D:/2022/3W/Oxford/SPCA_3/" + METHOD  + str(n_comp))            
            text_file = open("D:/2022/3W/Oxford/SPCA_3/" + METHOD  + str(n_comp) + "/Query"+str(q)+ ".txt","w")
            for t in range(len(b)):
                text_file.write(b[t][0] + '\n')
            text_file.close()

