import numpy as np
import os
import scipy.linalg 
from sympy import Symbol
from numpy.linalg import inv,det
import math as m
import cv2

query_name = np.load('Eva_NameList.npy')
Max_Cm = np.load('Max_Cm.npy')
Max_Cn = np.load('Max_Cn.npy')
Max_Result = np.load('Max_Result.npy')
Max_ResultM = np.load('Max_ResultM.npy')
Max_MEV =  np.load('Max_MEV.npy')
Max_MEV = np.reshape(Max_MEV, (512,512))
Ave_Cm = np.load('Ave_Cm.npy')
Ave_Cn = np.load('Ave_Cn.npy')
Ave_Result = np.load('Ave_Result.npy')
Ave_ResultM = np.load('Ave_ResultM.npy')
Ave_MEV =  np.load('Ave_MEV.npy')
Ave_MEV = np.reshape(Ave_MEV, (512,512))
Std_Cm = np.load('Std_Cm.npy')
Std_Cn = np.load('Std_Cn.npy')
Std_Result = np.load('Std_Result.npy')
Std_ResultM = np.load('Std_ResultM.npy')
Std_MEV =  np.load('Std_MEV.npy')
Std_MEV = np.reshape(Std_MEV, (512,512))
Cm = np.concatenate((Max_Cm,Ave_Cm,Std_Cm),axis=0)
Cn = np.concatenate((Max_Cn,Ave_Cn,Std_Cn),axis=0)
            
if __name__ == '__main__':
    print('Evaluation_2')
 
    Ave_Rshape_p5 = np.load('Eva_Ave_Rshape_p5.npy')
    Max_Rshape_p5 = np.load('Eva_Max_Rshape_p5.npy')
    Std_Rshape_p5 = np.load('Eva_Std_Rshape_p5.npy')
    NameList = np.load('Eva_NameList.npy')
    
    e = [1536, 1250, 1000, 750, 500, 250, 100]
    
    for n_comp in e:
        CIm = []
        CIn = []
        all_x = []
        x = Symbol('x')
        for i in range(3):
            itemm = []
            itemn = []
            x_item = []
            for j in range(512):
                CM = [[1,Cm[i,j]],[Cm[i,j],1]]
                CN = [[1,Cn[i,j]],[Cn[i,j],1]]
                a = Cn[i,j]-Cm[i,j]
                b = Cm[i,j]-Cn[i,j]
                c = 1-Cm[i,j]**2
                deltaCm = 1-Cm[i,j]**2
                deltaCn = 1-Cn[i,j]**2
                alpha = ((a**2)/c)*np.log(deltaCm/deltaCn)
                belta = ((2*(a**2)/c))-((2*Cm[i,j]*b)/c)*np.log(deltaCm/deltaCn)
                gamma = ((2*Cm[i,j]*a)/c) - np.log(deltaCm/deltaCn)
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
        
        CIm = np.reshape(CIm,(3,512))
        Ind = []  
        Sec_Cm = []
        Sec_Cn = []    
        dic = {}  
        for i in range(3):
            for j in range(512):
                dic[CIm[i,j]] = [i,j]
        M = sorted(dic.items(), key=lambda t: t[0],reverse = True)
        
        Ind = []
        a = 0
        b = 0
        c = 0
        d = 0
        for n in range(n_comp):
            Ind.append(M[n][1])
            Sec_Cm.append(Cm[M[n][1][0],M[n][1][1]])
            Sec_Cn.append(Cn[M[n][1][0],M[n][1][1]])
            if M[n][1][0] == 0:
                a = a + 1
            if M[n][1][0] == 1:
                b = b + 1
            if M[n][1][0] == 2:
                c = c + 1
            if M[n][1][0] == 3:
                d = d + 1
        np.save('Ind.npy',Ind)
        np.save('Sec_Cm.npy',Sec_Cm)
        np.save('Sec_Cn.npy',Sec_Cn)
        Ind = np.load('Ind.npy')
        #Ind = np.array(np.reshape(Ind,(1000,2)))
        Channel = Ind[:,0]
        Individual = Ind[:,1]
        
        L = len(Ind)
        #Ind = np.reshape(Ind,(1,L))
    
        Ave = Ave_Rshape_p5
        Max = Max_Rshape_p5
        Std = Std_Rshape_p5
        All_S_Transformed = []
        Max_AA_1_2 = scipy.linalg.fractional_matrix_power(Max_Result, -0.5)
        Ave_AA_1_2 = scipy.linalg.fractional_matrix_power(Ave_Result, -0.5)
        Std_AA_1_2 = scipy.linalg.fractional_matrix_power(Std_Result, -0.5)
        for mm in range(len(Ave_Rshape_p5)):        
            
            Transformed_Max = np.dot(np.transpose(Max_MEV),np.dot(Max_AA_1_2 ,Max[mm]))
            Transformed_Max = np.reshape(Transformed_Max,(1,512))
            Transformed_Ave = np.dot(np.transpose(Ave_MEV),np.dot(Ave_AA_1_2 ,Ave[mm]))
            Transformed_Ave = np.reshape(Transformed_Ave,(1,512))
            Transformed_Std = np.dot(np.transpose(Std_MEV),np.dot(Std_AA_1_2 ,Std[mm]))
            Transformed_Std = np.reshape(Transformed_Std,(1,512))


            Transformed = np.concatenate((Transformed_Max, Transformed_Ave,Transformed_Std), axis=0)
            S_Transformed = Transformed[Channel,Individual] 
            S_Transformed = np.reshape(S_Transformed,(1,n_comp))
            S_B = np.sqrt(np.sum(S_Transformed**2))
            S_norm_B = S_Transformed/S_B
            Transformed_Ave = np.nan_to_num(S_norm_B)   
            All_S_Transformed.append(S_Transformed)
            
        
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
                P = 0
                Xi = np.reshape(X,(L,1))
                Yi = np.reshape(Y,(L,1))          
                for r in range(L):  
                    CM= Sec_Cm[r]
                    CN= Sec_Cn[r]
                    
                    CV_X = Xi[r,0]
                    CV_Y = Yi[r,0]
                    
                    A = (CV_X**2) + (CV_Y**2) - 2*CV_X*CV_Y*CM   
                    B = (CV_X**2) + (CV_Y**2) - 2*CV_X*CV_Y*CN   
                    C = 2-2*(CM**2)
                    D = 2-2*(CN**2)
                    E = m.log(m.sqrt(D/C))
                    F = -(A/C) + (B/D) + E
                          
                    P = P + F 
                    
                dic[All_Name[k][:-4]] = P
            b = sorted(dic.items(), key=lambda d: d[1],reverse=True)
            if not os.path.exists("D:/2022/12W/Oxford/G-CCCA/" + str(n_comp)):
                os.makedirs("D:/2022/12W/Oxford/G-CCCA/" + str(n_comp))            
            text_file = open("D:/2022/12W/Oxford/G-CCCA/" + str(n_comp) + "/Query"+str(q)+ ".txt","w")
            for t in range(len(b)):
                text_file.write(b[t][0] + '\n')
            text_file.close()

