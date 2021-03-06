import numpy as np
import os
import scipy.linalg 
from sympy import Symbol
from numpy.linalg import inv,det
import math as m


METHOD = 'Std'
Ave_MEV =  np.load(METHOD + '_MEV.npy')
Ave_MEV = np.reshape(Ave_MEV, (512,512))
Ave_Result = np.load(METHOD + '_Result.npy')
Ave_ResultM = np.load(METHOD + '_ResultM.npy')
query_name = np.load('Eva_NameList.npy')
Cm = np.load(METHOD + '_Cm.npy')
Cn = np.load(METHOD + '_Cn.npy')

            
if __name__ == '__main__':
    print('Evaluation_2')
 
    Ave_Rshape_p5 = np.load('Eva_'+ METHOD +'_Rshape_p5.npy')
    NameList = np.load('Eva_NameList.npy')
    
    e = [512,450,400,300,200,100,50,25]
    for n_comp in e:        
        x = Symbol('x')
        CIm = []
        CIn = []
        all_x = []
        itemm = []
        itemn = []
        x_item = []
        for j in range(512):
            CM = [[1,Cm[0,j]],[Cm[0,j],1]]
            CN = [[1,Cn[0,j]],[Cn[0,j],1]]
            a = Cn[0,j]-Cm[0,j]
            b = Cm[0,j]-Cn[0,j]
            c = 1-Cm[0,j]**2
            deltaCm = 1-Cm[0,j]**2
            deltaCn = 1-Cn[0,j]**2
            alpha = ((a**2)/c)*np.log(deltaCm/deltaCn)
            belta = ((2*(a**2)/c))-((2*Cm[0,j]*b)/c)*np.log(deltaCm/deltaCn)
            gamma = ((2*Cm[0,j]*a)/c) - np.log(deltaCm/deltaCn)
            x = (-belta+np.sqrt(belta**2 - 4*alpha*gamma))/(2*alpha)
            x_item.append(x)
            X = inv(x*inv(CM) + (1-x)*inv(CN))
            deltax = det(X)
            CI_itemm = 0.5*(np.log(deltaCm/deltax)) + 0.5*np.trace(np.dot(inv(CM),X)) - 1
            CI_itemn = 0.5*(np.log(deltaCn/deltax)) + 0.5*np.trace(np.dot(inv(CN),X)) - 1
            itemm.append(CI_itemm)
            itemn.append(CI_itemn)
        CIm.append(itemm)
        CIn.append(itemn)
        all_x.append(x_item)    
        
        CIm = np.reshape(CIm,(1,512))
        Ind = []  
        Sec_Cm = []
        Sec_Cn = []    
        dic = {}  
        for j in range(512):
            dic[CIm[0,j]] = [j]
        M = sorted(dic.items(), key=lambda t: t[0],reverse = True)
        
        Ind = []
        for n in range(n_comp):
            Ind.append(M[n][1])
            Sec_Cm.append(Cm[0,M[n][1][0]])
            Sec_Cn.append(Cn[0,M[n][1][0]])
        np.save('Ind.npy',Ind)
        np.save('Sec_Cm.npy',Sec_Cm)
        np.save('Sec_Cn.npy',Sec_Cn)
        
        L = len(Ind)
        Ind = np.reshape(Ind,(1,L))
    
        Ave = Ave_Rshape_p5
        All_S_Transformed = []
        S_Ave_MEV = Ave_MEV[:,Ind[0]]
        Ave_AA_1_2 = scipy.linalg.fractional_matrix_power(Ave_Result, -0.5)
        for mm in range(len(Ave_Rshape_p5)):       
            Transformed_Ave = np.dot(np.transpose(S_Ave_MEV),np.dot(Ave_AA_1_2 ,Ave[mm]))
            Transformed_Ave = np.reshape(Transformed_Ave,(1,n_comp))
            std_S_B = np.sqrt(np.sum(Transformed_Ave**2))
            std_norm_B = Transformed_Ave/std_S_B
            std_norm_B = np.nan_to_num(std_norm_B)
            All_S_Transformed.append(std_norm_B)
        
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
                Y = All[k]
                Xi = np.reshape(X,(n_comp,1))
                Yi = np.reshape(Y,(n_comp,1))          
                P = np.dot(np.transpose(Xi),Yi)               
                dic[All_Name[k][:-4]] = R
            b = sorted(dic.items(), key=lambda d: d[1],reverse=True)
            if not os.path.exists("./Test_results/Oxford/S-CCA/" + METHOD + '_N'  + str(n_comp)):
                os.makedirs("./Test_results/Oxford/S-CCA/" + METHOD  + '_N'  + str(n_comp))            
            text_file = open("./Test_results/Oxford/S-CCA/" + METHOD  + '_N'  + str(n_comp) + "/Query"+str(q)+ ".txt","w")
            for t in range(len(b)):
                text_file.write(b[t][0] + '\n')
            text_file.close()

