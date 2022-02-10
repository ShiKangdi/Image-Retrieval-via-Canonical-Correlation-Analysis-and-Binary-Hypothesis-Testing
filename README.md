# Image-Retrieval-via-Canonical-Correlation-Analysis-and-Binary-Hypothesis-Testing
Codes for proposed approach in "Image Retrieval via Canonical Correlation Analysis and Binary Hypothesis Testing

Step 1:

GeneratePairs.py: This file generates the matching and non-matching pairs.

Training_Data_preparation.py: For feature extraction on training dataset, it saves each type of features as ''Max_Rshape_P5.npy', 'Ave_Rshape_P5.npy', 'Std_Rshape_P5.npy', and also save names for each training sample as 'NameList.npy'

Test_Data_preparation.py: For feature extraction on training dataset, it saves each type of features as ''Eva_Max_Rshape_P5.npy', 'Eva_Ave_Rshape_P5.npy', 'Eva_Std_Rshape_P5.npy', and also save names for each training sample as 'Eva_NameList.npy'.

Step 2:

CCA_preparation.py: For each type of features from training dataset, It respectively save matching and non-matching coefficients as "Max_Cm.py", "Ave_Cn.py", "Std_Cm.py", "Ave_Cn.py", "Ave_Cm.py", "Std_Cn.py", and canoncial vectors as "Max_MEV.py", "Ave_MEV.py", "Std_MEV.py". On the otherhand, it saves the training data as "Max_Reshape.npy","Ave_Reshape.npy","Std_Reshape.npy" after centralized and normalization.

LDA_preparation.py: For each type of features from training dataset, It respectively save between-class scatter matrix as "max_Sb.py","ave_Sb.py","std_Sb.py", and within-class scatter matrix as "max_Sw.py","ave_Sw.py","std_Sw.py". It needs the feature means before and after centralization and normalization. These files can be obtained from "CCA_preparation.py" and "Training_Data_preparation.py".

SPCAMatrix.py: The Laplacian matrix used for SPCA is calculated in this file. The order of each image is based on generated matching pairs.

Step 3:

Evaluation_LDA.py: It evaluates the image retrieval performance of LDA. It needs between-class scatter matrix("max_Sb.py","ave_Sb.py","std_Sb.py") and within-class scatter matrix("max_Sw.py","ave_Sw.py","std_Sw.py") from the outputs of LDA_preparation.py. The input file "Eva_NameList.npy" which can be obtained from the output of "Test_Data_preparation.py", and it contains the name of each test data which is used for generating the retrieval results in txt format. 

Evaluation_PCA.py: It evaluates the image retrieval performance of PCA. It needs the centralized and normalized features("Max_Reshape.npy","Ave_Reshape.npy","Std_Reshape.npy") from training set as inputs.  "Eva_NameList.npy" is also needed to generating the retrieval results in txt format.

Evaluation_SPCA.py: It evaluates the image retrieval performance of SPCA. It needs the centralized and normalized features("Max_Reshape.npy","Ave_Reshape.npy","Std_Reshape.npy") from training set as inputs, and Laplacian matrix from the output of "SPCAMatrix.py". "Eva_NameList.npy" is also needed to generating the retrieval results in txt format.

Evaluation_G-CCA.py: It evaluates the image retrieval performance of G-CCA. It needs matching and non-matching coefficients "Max_Cm.py", "Ave_Cn.py", "Std_Cm.py", "Ave_Cn.py", "Ave_Cm.py", "Std_Cn.py", and canoncial vectors "Max_MEV.py", "Ave_MEV.py", "Std_MEV.py" from training set as inputs. "Eva_NameList.npy" is also needed to generating the retrieval results in txt format.

Evaluation_S-CCA.py: It evaluates the image retrieval performance of S-CCA. It needs matching and non-matching coefficients "Max_Cm.py", "Ave_Cn.py", "Std_Cm.py", "Ave_Cn.py", "Ave_Cm.py", "Std_Cn.py", and canoncial vectors "Max_MEV.py", "Ave_MEV.py", "Std_MEV.py" from training set as inputs. "Eva_NameList.npy" is also needed to generating the retrieval results in txt format.

test_Map.m: It tests the Map based on the retrieval results in txt format. 
