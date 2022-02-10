# Image-Retrieval-via-Canonical-Correlation-Analysis-and-Binary-Hypothesis-Testing
Codes for proposed approach in "Image Retrieval via Canonical Correlation Analysis and Binary Hypothesis Testing

Step 1:

GeneratePairs.py: This file generates the matching and non-matching pairs.

Training_Data_preparation.py: For feature extraction on training dataset, it saves each type of features as ''Max_Rshape_P5.npy', 'Ave_Rshape_P5.npy', 'Std_Rshape_P5.npy', and also save names for each training sample as 'NameList.npy'

Test_Data_preparation.py: For feature extraction on training dataset, it saves each type of features as ''Eva_Max_Rshape_P5.npy', 'Eva_Ave_Rshape_P5.npy', 'Eva_Std_Rshape_P5.npy', and also save names for each training sample as 'Eva_NameList.npy'.

Step 2:

CCA_preparation.py: For each type of features from training dataset, It respectively save matching and non-matching coefficients as "Max_Cm.py", "Ave_Cn.py", "Std_Cm.py", "Ave_Cn.py", "Ave_Cm.py", "Std_Cn.py", and canoncial vectors as "Max_MEV.py", "Ave_MEV.py", "Std_MEV.py".

LDA_preparation.py: For each type of features from training dataset, It respectively save between-class scatter matrix as "max_Sb.py","ave_Sb.py","std_Sb.py", and within-class scatter matrix as "max_Sw.py","ave_Sw.py","std_Sw.py". It needs the feature means before and after centralization and normalization. These files can be obtained from CCA_preparation.py.

SPCAMatrix.py: the Laplacian matrix used for SPCA is calculated in this file. The order of each image is based on generated matching pairs.

Step 3:

Evaluation_LDA.py: It evaluates the image retrieval performance of LDA. It needs between-class scatter matrix and within-class scatter matrix from the outputs of LDA_preparation.py. The input file "Eva_NameList.npy" which can be obtained from the output of "Test_Data_preparation.py", and it contains the name of each test data which is used for generating the retrieval results in txt format.

Evaluation_PCA.py: It evaluates the image retrieval performance of PCA.

Evaluation_SPCA.py: It evaluates the image retrieval performance of SPCA.

Evaluation_G-CCA.py: It evaluates the image retrieval performance of G-CCA.

Evaluation_S-CCA.py: It evaluates the image retrieval performance of S-CCA.

test_Map.m: It test the Map based on the retrieval results in txt files. 
