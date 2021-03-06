# Image-Retrieval-via-Canonical-Correlation-Analysis-and-Binary-Hypothesis-Testing
Codes for proposed approach in "Image Retrieval via Canonical Correlation Analysis and Binary Hypothesis Testing

Step 1: Generate matching and non-matching pairs, and extract image feature vectors from VGG16.

GeneratePairs.py: This file generates the matching and non-matching pairs.

Training_Data_preparation.py: For feature extraction on training dataset, it saves each type of features as ''Max_Rshape_P5.npy', 'Ave_Rshape_P5.npy', 'Std_Rshape_P5.npy', and also save names for each training sample as 'NameList.npy'

Test_Data_preparation.py: For feature extraction on testing dataset, it saves each type of features as ''Eva_Max_Rshape_P5.npy', 'Eva_Ave_Rshape_P5.npy', 'Eva_Std_Rshape_P5.npy', and also save names for each training sample as 'Eva_NameList.npy'.



Step 2: Prepare coefficients and matrix(vectors) for each method.

CCA_preparation.py: For each type of features from training dataset, It respectively save matching and non-matching coefficients as "Max_Cm.py", "Ave_Cn.py", "Std_Cm.py", "Ave_Cn.py", "Ave_Cm.py", "Std_Cn.py", canoncial vectors as "Max_MEV.py", "Ave_MEV.py", "Std_MEV.py", and the auto-correction matrix as "Max_Result.npy", "Ave_Result.npy", "Std_Result.npy". Besides, it saves the training data as "Max_Reshape.npy","Ave_Reshape.npy","Std_Reshape.npy" after centralized and normalization. 

LDA_preparation.py: For each type of features from training dataset, It respectively save between-class scatter matrix as "max_Sb.py","ave_Sb.py","std_Sb.py", and within-class scatter matrix as "max_Sw.py","ave_Sw.py","std_Sw.py". It needs the feature means ("Max_ResultM.npy","Ave_ResultM.npy","Std_ResultM.npy") before centralization and normalization as inputs, and feature means after centralization and normalization that is calculated based on associated features ("Max_Reshape.npy","Ave_Reshape.npy","Std_Reshape.npy"). These files can be obtained from the output "CCA_preparation.py" and "Training_Data_preparation.py".

SPCAMatrix.py: It generates the Laplacian matrix 'Laplcianmatrix.npy' which is used for Supervised PCA. The order of labels is based on generated matching pairs in "GeneratePairs.py".




Step 3: Evaluation

Evaluation_LDA.py: It evaluates the image retrieval performance of LDA. It needs between-class scatter matrix("max_Sb.py","ave_Sb.py","std_Sb.py") and within-class scatter matrix("max_Sw.py","ave_Sw.py","std_Sw.py") from the outputs of LDA_preparation.py. The input file "Eva_NameList.npy" which can be obtained from the output of "Test_Data_preparation.py", and it contains the name of each test data which is used to generate the retrieval results in txt format. 

Evaluation_PCA.py: It evaluates the image retrieval performance of PCA. It needs the centralized and normalized features("Max_Reshape.npy","Ave_Reshape.npy","Std_Reshape.npy") from training set as inputs.  Besides, the "Eva_NameList.npy" is used to generate the retrieval results in txt format.

Evaluation_SPCA.py: It evaluates the image retrieval performance of SPCA. It needs the centralized and normalized features("Max_Reshape.npy","Ave_Reshape.npy","Std_Reshape.npy") from training set as inputs, and Laplacian matrix "SPCAMatrix.py". Besides, the "Eva_NameList.npy" is used to generating the retrieval results in txt format.

Evaluation_G-CCA.py: It evaluates the image retrieval performance of G-CCA. It needs matching and non-matching coefficients "Max_Cm.py", "Ave_Cn.py", "Std_Cm.py", "Ave_Cn.py", "Ave_Cm.py", "Std_Cn.py", canoncial vectors "Max_MEV.py", "Ave_MEV.py", "Std_MEV.py" and auto-correlation matrix "Max_Result.npy", "Ave_Result.npy", "Std_Result.npy" from training set as inputs.  Besides, the "Eva_NameList.npy" is used to generate the retrieval results in txt format.

Evaluation_S-CCA.py: It evaluates the image retrieval performance of S-CCA. It needs matching and non-matching coefficients "Max_Cm.py", "Ave_Cn.py", "Std_Cm.py", "Ave_Cn.py", "Ave_Cm.py", "Std_Cn.py", and canoncial vectors "Max_MEV.py", "Ave_MEV.py", "Std_MEV.py" from training set as inputs. "Eva_NameList.npy" is also needed to generating the retrieval results in txt format.

gnd_oxford5k.mat, gnd_paris6k.mat, gnd_roxford5k.mat, gnd_rparis6k.mat: Mat files that contain postive labels for calculating Map.

score_ap_from_ranks.m: This file calculates the average precision for each query image.

compute_map_r.m: This file calculate the Map based on retrieval results of all query images.

test_Map.m: It tests the Map based on the retrieval results in txt format. "gnd_oxford5k.mat","gnd_paris6k.mat","gnd_roxford5k.mat","gnd_rparis6k.mat", and "score_ap_from_ranks.m" are used in this file.



