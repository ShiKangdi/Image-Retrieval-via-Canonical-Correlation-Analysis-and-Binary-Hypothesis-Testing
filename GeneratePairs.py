import os
from shutil import copyfile
import random
import shutil
from nt import chdir

Source = 'E:/New/instre/S2' # Your dictionary for original dataset. Images for same class is put in same folder
PairA = 'E:/New/instre/S2Pair/Matching/PairA' # Your dictionary for One side of Matching pairs
PairB = 'E:/New/instre/S2Pair/Matching/PairB' # Your dictionary for Another side of Matching pairs
for filenamea in os.listdir(Source):
    M = Source + '/' + str(filenamea)
    files = os.listdir(M)
    R = len(files)//2
    for j in range(0,R):
        A = Source + '/' + str(filenamea) + '/' + files[j]
        DA = PairA +'/'+ str(filenamea)
        if not os.path.exists(DA):
            os.makedirs(DA)
        copyfile(A, DA + '/'+  files[j])
        
    for k in range(R,len(files)):
        B = Source + '/' + str(filenamea) + '/' + files[k]
        DB = PairB +'/'+ str(filenamea)
        if not os.path.exists(DB):
            os.makedirs(DB)
        copyfile(B, DB + '/' + files[k])
        
PairA = 'E:/New/instre/S2Pair/NonMatching/PairA' # Your dictionary for One side of Non-Matching pairs
PairB = 'E:/New/instre/S2Pair/NonMatching/PairB' # Your dictionary for Another side of Temporary Non-Matching pairs. This should be empty after non-matching pairs generated.
Real = 'E:/New/instre/S2Pair/NonMatching/RealPairB' # Your dictionary for Another side of Real Non-Matching pairs
for filenamea in os.listdir(Source):
    M = Source + '/' + str(filenamea)
    files = os.listdir(M)
    R = len(files)//2
    for q in range(0,R):
        A = Source + '/' + str(filenamea) + '/' + files[q]
        DA = PairA+'/'+ str(filenamea)
        if not os.path.exists(DA):
            os.makedirs(DA)
        copyfile(A, DA + '/'+  files[q])
                
    for w in range(R,len(files)):
        B = Source + '/' + str(filenamea) + '/' + files[w]
        DB = PairB +'/'+ str(filenamea)
        if not os.path.exists(DB):
            os.makedirs(DB)
        copyfile(B, DB + '/' + files[w])
        

K = os.listdir(PairB)
for item in K: 
    HH = len(os.listdir(PairA +'/'+item)) 
    M = K.index(item) 
    N = K.index(item) 
    for i in range(HH):
        flag = 1
        while (flag ==1) : 
            N = random.randint(0,len(K)-1)
            GG = os.listdir(PairB+'/'+K[N])
            if (M != N) and (len(GG) != 0):
                flag = 0
        TheFile = random.randint(0, len(GG)-1)
        if not os.path.exists(Real+'/'+K[M]):
            os.makedirs(Real+'/'+K[M])
        shutil.move(PairB+'/'+K[N]+'/'+ GG[TheFile],Real+'/'+K[M]+'/'+ GG[TheFile])
    

