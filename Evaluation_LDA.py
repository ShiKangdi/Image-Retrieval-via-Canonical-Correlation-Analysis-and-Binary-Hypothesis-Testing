import numpy as np
import os
import scipy.linalg 
from numpy.linalg import inv,det
import math as m

METHOD = 'Max'
Ave_Sb = np.load(METHOD + '_Sb.npy')
Ave_Sw = np.load(METHOD + '_Sw.npy')
LDA = np.dot(inv(Ave_Sw),Ave_Sb)
Eigval,Eigvec = eig(LDA)
TrEigval = np.sort(Eigval)[::-1]
EigvecInd = np.argsort(Eigval)[::-1]
TrEigvec = Eigvec[:,EigvecInd]
query_name = np.load('Eva_NameList.npy')

if __name__ == '__main__':
    print('Evaluation_2')
 
    Max_Rshape_p5 = np.load('Eva_'+ METHOD +'_Rshape_p5.npy')
    NameList = np.load('Eva_NameList.npy')
    
    e = [512,450,400,300,200,100,50,25]
    for n_comp in e:
        R_LDA = TrEigvec[:,:n_comp]
        All_S_Transformed = []
        for mm in range(len(Max_Rshape_p5)):
            Max_DPB = np.dot(np.transpose(R_LDA),Max_Rshape_p5[mm])
            Max_DPB = np.nan_to_num(Max_DPB)
            Max_S_B = np.sqrt(np.sum(Max_DPB**2))
            Max_norm_B = Max_DPB/Max_S_B
            Max_norm_B = np.nan_to_num(Max_norm_B)
            All_S_Transformed.append(Max_norm_B)
        np.save('Eva_Trans_Ave_Rshape_P5.npy',All_S_Transformed)  
#    
        Query = All_S_Transformed[:55]
        All = All_S_Transformed[55:]
        All_Name = NameList[55:]
    
        np.save('Sec_All.npy',All)
        np.save('Sec_Query.npy',Query)
        np.save('Sec_All_Name.npy',All_Name)

        All = np.load('Sec_All.npy')
        All_name = np.load('Sec_All_name.npy')
        Query = np.load('Sec_Query.npy')
        
        O = len(All)
        for q in range(55):
            dic = {}
            X = Query[q]
            a = []
            for k in range(n_comp):
                print([q,k])
                Y = All[k]
                Xi = np.reshape(X,(n_comp,1))
                Yi = np.reshape(Y,(n_comp,1))           
                P = np.dot(np.transpose(Xi),Yi)               
                dic[All_name[k][:-4]] = P
                a.append(P)
            b = sorted(dic.items(), key=lambda d: d[1],reverse=True)
            if not os.path.exists("./Test_results/Oxford/LDA/" + METHOD  + str(n_comp)):
                os.makedirs("./Test_results/Oxford/LDA/" + METHOD  + str(n_comp))            
            text_file = open("./Test_results/Oxford/LDA/" + METHOD  + str(n_comp) + "/Query"+str(q)+ ".txt","w")
            for t in range(len(b)):
                text_file.write(b[t][0] + '\n')
            text_file.close()

